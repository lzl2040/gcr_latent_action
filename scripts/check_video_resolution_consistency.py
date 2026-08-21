#!/usr/bin/env python
"""检查 LeRobot 数据集的视频分辨率是否自洽，定位 torchcodec 批量解码报错。

背景
----
训练时可能看到:

    Failed to read robomind_franka_3rgb[127063]:
        Expected pre-allocated tensor of shape 480x640x3, got [720, 1280, 3]

这条消息来自 torchcodec 的 CpuDeviceInterface.cpp。注意它的方向容易读反:

    "Expected <实际解码出来的帧尺寸>, got <按流元数据预分配的张量尺寸>"

即 **元数据说 720x1280，但真正解出来的帧是 480x640**。torchcodec 的批量 API
(`get_frames_at`) 会先按流元数据一次性预分配 (N, H, W, 3)，而 mp4 的流元数据只
记录第一段的分辨率。所以只要一个 mp4 内部混了两种分辨率，批量解码必炸（单帧 API
反而不会，因为它按实际帧尺寸分配）。

这在 v3.0 布局下尤其容易发生: v3.0 把多个 episode 拼进同一个 mp4，只要其中有
分辨率不同的 episode，拼出来的文件就是变分辨率的。v2.1 一个 episode 一个 mp4，
同样的数据反而不会报错——所以"本地能跑、集群报错"往往就是布局差异导致的。

用法
----
    conda activate lerobot_v2

    # 完整检查（推荐）
    python scripts/check_video_resolution_consistency.py \
        --root /Data/lerobot_data_ort6d/v30/RoboMind_full/franka_3rgb

    # 把训练日志里报错的帧号反查到 episode / 视频文件
    python scripts/check_video_resolution_consistency.py \
        --root ... --frames 127063 194891

    # 逐帧穷举扫描（用 ffprobe，最慢但最确定，能发现采样漏掉的分辨率突变）
    python scripts/check_video_resolution_consistency.py --root ... --exhaustive

    # 冒烟测试: 只看前 200 个视频
    python scripts/check_video_resolution_consistency.py --root ... --limit 200
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# 元数据读取（兼容 v2.1 / v3.0 两种布局）
# ============================================================================

def load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"找不到 {info_path}，--root 是否指向数据集根目录？")
    return json.loads(info_path.read_text())


def video_keys_of(info: dict) -> list[str]:
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]


def declared_shapes(info: dict) -> dict[str, tuple[int, int]]:
    """info.json 里声明的 (height, width)，按 video key。"""
    out = {}
    for k, v in info.get("features", {}).items():
        if v.get("dtype") != "video":
            continue
        shape = v.get("shape")
        if shape and len(shape) >= 2:
            out[k] = (int(shape[0]), int(shape[1]))
    return out


def load_episodes(root: Path, info: dict, keys: list[str]) -> list[dict]:
    """统一读出 episode 列表: episode_index / start / end / videos{key: path}。

    v2.1 从 meta/episodes.jsonl + video_path 模板推路径;
    v3.0 从 meta/episodes/**/*.parquet 的 videos/<key>/{chunk,file}_index 推路径。
    读不到就返回空列表，不影响分辨率扫描本身。
    """
    template = info.get("video_path", "")
    v21_jsonl = root / "meta" / "episodes.jsonl"
    v30_meta = sorted((root / "meta" / "episodes").glob("**/*.parquet"))

    episodes: list[dict] = []
    if v30_meta:
        import pyarrow.parquet as pq

        for f in v30_meta:
            table = pq.read_table(f)
            cols = table.column_names
            data = table.to_pydict()
            n = len(data["episode_index"])
            for i in range(n):
                rec = {
                    "episode_index": int(data["episode_index"][i]),
                    "start": int(data["dataset_from_index"][i]),
                    "end": int(data["dataset_to_index"][i]),
                    "videos": {},
                }
                for key in keys:
                    ck, fk = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
                    if ck in cols and fk in cols:
                        rec["videos"][key] = template.format(
                            video_key=key,
                            chunk_index=int(data[ck][i]),
                            file_index=int(data[fk][i]),
                        )
                episodes.append(rec)
    elif v21_jsonl.is_file():
        chunks_size = int(info.get("chunks_size", 1000))
        acc = 0
        raw = [json.loads(line) for line in v21_jsonl.read_text().splitlines() if line.strip()]
        for rec in sorted(raw, key=lambda r: r["episode_index"]):
            ep = int(rec["episode_index"])
            length = int(rec["length"])
            episodes.append({
                "episode_index": ep,
                "start": acc,
                "end": acc + length,
                "videos": {
                    key: template.format(
                        episode_chunk=ep // chunks_size, video_key=key, episode_index=ep
                    )
                    for key in keys
                },
            })
            acc += length

    episodes.sort(key=lambda r: r["start"])
    return episodes


# ============================================================================
# 单文件探测（跑在 worker 进程里）
# ============================================================================

def probe_stream(path: str) -> tuple[int, int] | None:
    """ffprobe 读流级 (h, w) —— 这正是 torchcodec 批量 API 用来预分配的尺寸。"""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        w, h = proc.stdout.strip().split(",")[:2]
        return int(h), int(w)
    except Exception:
        return None


def probe_frames_ffprobe(path: str) -> list[tuple[int, int]] | None:
    """ffprobe 逐帧读 (h, w)，返回去重后的集合。慢，但能确定地发现分辨率突变。"""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=width,height", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            return None
        seen = []
        for line in proc.stdout.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            w, h = line.split(",")[:2]
            dims = (int(h), int(w))
            if dims not in seen:
                seen.append(dims)
        return seen
    except Exception:
        return None


def probe_decode(path: str, n_samples: int) -> dict:
    """用 torchcodec 复现训练时的批量解码，抓到的异常就是训练里看到的那条。

    失败后再逐帧单独解码（单帧 API 按实际帧尺寸分配，不会炸），
    把每个采样点的真实分辨率报出来，从而定位分辨率在哪里发生了变化。
    """
    out = {"path": path, "ok": True, "error": None, "meta_hw": None,
           "num_frames": None, "frame_hw": {}}
    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(path, seek_mode="approximate")
        meta = decoder.metadata
        out["meta_hw"] = (int(meta.height), int(meta.width))
        num_frames = int(meta.num_frames or 0)
        out["num_frames"] = num_frames
        if num_frames <= 0:
            out["ok"] = False
            out["error"] = "num_frames <= 0"
            return out

        step = max(1, num_frames // max(1, n_samples))
        indices = sorted({min(i, num_frames - 1) for i in range(0, num_frames, step)})
        indices = indices[:n_samples] or [0]

        try:
            # 批量 API: 按流元数据预分配，变分辨率时会在这里抛错
            decoder.get_frames_at(indices=indices)
        except Exception as exc:
            out["ok"] = False
            out["error"] = f"{type(exc).__name__}: {exc}"
            # 单帧 API 逐个解，找出分辨率变化的位置
            for i in indices:
                try:
                    frame = decoder[i]
                    out["frame_hw"][i] = (int(frame.shape[-2]), int(frame.shape[-1]))
                except Exception as single_exc:  # noqa: BLE001
                    out["frame_hw"][i] = f"decode failed: {single_exc}"
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _probe_one(args):
    path, n_samples, exhaustive = args
    rec = {"path": path, "stream_hw": probe_stream(path)}
    if n_samples > 0:
        rec.update({k: v for k, v in probe_decode(path, n_samples).items() if k != "path"})
    if exhaustive:
        rec["all_frame_hw"] = probe_frames_ffprobe(path)
    return rec


# ============================================================================
# 主流程
# ============================================================================

def key_of(path: Path, root: Path, keys: list[str]) -> str:
    parts = set(path.relative_to(root).parts)
    for k in keys:
        if k in parts:
            return k
    return "<unknown>"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检查数据集视频分辨率是否自洽（定位 torchcodec 预分配报错）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", type=Path,
                        default=Path("/Data/lerobot_data_ort6d/v30/RoboMind_full/franka_3rgb"),
                        help="数据集根目录（含 meta/ videos/）")
    parser.add_argument("--keys", nargs="*", default=None,
                        help="只检查这些 video key（默认全部）")
    parser.add_argument("--workers", type=int, default=16, help="并行进程数")
    parser.add_argument("--sample-frames", type=int, default=16,
                        help="每个视频用 torchcodec 采样解码多少帧（0 = 跳过解码，只看 ffprobe）")
    parser.add_argument("--exhaustive", action="store_true",
                        help="用 ffprobe 逐帧扫描每个视频（最慢最确定；采样可能漏掉分辨率突变）")
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="把训练日志里报错的全局帧号反查到 episode / 视频文件")
    parser.add_argument("--limit", type=int, default=None, help="只检查前 N 个视频文件")
    parser.add_argument("--json", type=Path, default=None, help="把完整报告写到该 JSON 文件")
    args = parser.parse_args()

    root = args.root.resolve()
    info = load_info(root)
    all_keys = video_keys_of(info)
    keys = args.keys or all_keys
    declared = declared_shapes(info)

    print("=" * 78)
    print(f"数据集: {root}")
    print(f"codebase_version: {info.get('codebase_version')}   fps: {info.get('fps')}")
    print(f"episodes: {info.get('total_episodes')}   frames: {info.get('total_frames')}")
    print(f"video_path: {info.get('video_path')}")
    print(f"video keys: {all_keys}")
    for k in keys:
        if k in declared:
            print(f"  info.json 声明 {k}: {declared[k][0]}x{declared[k][1]} (HxW)")
    print("=" * 78)

    videos = sorted(p for p in (root / "videos").rglob("*.mp4")
                    if key_of(p, root, all_keys) in keys)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print("!! videos/ 下没找到任何 mp4")
        return 1
    print(f"\n待检查视频文件: {len(videos)}")
    if args.exhaustive:
        print("   (--exhaustive: 逐帧扫描，会比较慢)")

    tasks = [(str(p), args.sample_frames, args.exhaustive) for p in videos]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_probe_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 500 == 0 or i == len(tasks):
                print(f"   ... {i}/{len(tasks)}")

    by_path = {r["path"]: r for r in results}

    # ---- 汇总 1: 每个 key 的分辨率分布 ----
    print("\n" + "=" * 78)
    print("【1】各 video key 的流级分辨率分布")
    print("=" * 78)
    per_key = defaultdict(Counter)
    for r in results:
        per_key[key_of(Path(r["path"]), root, all_keys)][r["stream_hw"]] += 1
    mixed_keys = []
    for k in sorted(per_key):
        print(f"\n  {k}   (info.json 声明 "
              f"{declared.get(k, ('?', '?'))[0]}x{declared.get(k, ('?', '?'))[1]})")
        for dims, count in per_key[k].most_common():
            tag = ""
            if dims is None:
                tag = "  <- ffprobe 读取失败"
            elif k in declared and dims != declared[k]:
                tag = "  <- 与 info.json 声明不一致"
            shown = f"{dims[0]}x{dims[1]}" if dims else "None"
            print(f"      {shown:>12}  x {count}{tag}")
        real = [d for d in per_key[k] if d is not None]
        if len(real) > 1:
            mixed_keys.append(k)

    # ---- 汇总 2: 单个文件内部变分辨率（这才是训练报错的直接原因） ----
    print("\n" + "=" * 78)
    print("【2】单个 mp4 内部是否混了多种分辨率（torchcodec 批量解码报错的直接原因）")
    print("=" * 78)
    broken = [r for r in results if r.get("ok") is False]
    exhaustive_bad = [r for r in results
                      if r.get("all_frame_hw") and len(r["all_frame_hw"]) > 1]
    if not broken and not exhaustive_bad:
        scope = "逐帧穷举" if args.exhaustive else f"每文件采样 {args.sample_frames} 帧"
        print(f"  ✅ 没发现问题（检查范围: {scope}）")
    for r in broken:
        print(f"\n  ❌ {Path(r['path']).relative_to(root)}")
        print(f"     元数据: {r['meta_hw']}  帧数: {r['num_frames']}")
        print(f"     报错: {r['error']}")
        if r.get("frame_hw"):
            hist = Counter(str(v) for v in r["frame_hw"].values())
            print(f"     采样帧实际分辨率: {dict(hist)}")
    for r in exhaustive_bad:
        if r.get("ok") is not False:
            print(f"\n  ❌ {Path(r['path']).relative_to(root)} 逐帧扫描发现多种分辨率: "
                  f"{r['all_frame_hw']}")

    # ---- 汇总 3: 帧号反查 ----
    episodes = load_episodes(root, info, keys)
    if args.frames:
        print("\n" + "=" * 78)
        print("【3】报错帧号反查")
        print("=" * 78)
        if not episodes:
            print("  读不到 episode 元数据，无法反查")
        for g in args.frames:
            match = next((e for e in episodes if e["start"] <= g < e["end"]), None)
            if match is None:
                print(f"  frame {g}: 超出数据集范围")
                continue
            print(f"\n  frame {g} -> episode {match['episode_index']} "
                  f"(局部帧 {g - match['start']}, 区间 [{match['start']}, {match['end']}))")
            for k, rel in sorted(match["videos"].items()):
                rec = by_path.get(str(root / rel))
                dims = rec["stream_hw"] if rec else None
                shown = f"{dims[0]}x{dims[1]}" if dims else "?"
                flag = ""
                if k in declared and dims and dims != declared[k]:
                    flag = f"  <- 与声明 {declared[k][0]}x{declared[k][1]} 不一致"
                if rec and rec.get("ok") is False:
                    flag += "  <- 该文件解码失败"
                print(f"      {k:<40} {shown:>10}{flag}  {rel}")

    # ---- 结论 ----
    print("\n" + "=" * 78)
    print("【结论】")
    print("=" * 78)
    exit_code = 0
    if broken or exhaustive_bad:
        exit_code = 1
        print(f"  ❌ 有 {len(set([r['path'] for r in broken] + [r['path'] for r in exhaustive_bad]))} "
              f"个 mp4 内部混了多种分辨率。")
        print("     torchcodec 的批量 API 按流元数据预分配张量，遇到不同尺寸的帧就会抛")
        print("     'Expected pre-allocated tensor of shape ...'。必须重新编码这些视频，")
        print("     把整段统一到同一分辨率。")
    if mixed_keys:
        if exit_code == 0:
            exit_code = 2
        print(f"\n  ⚠️  这些 key 的不同文件之间分辨率不一致: {mixed_keys}")
        for k in mixed_keys:
            odd = defaultdict(list)
            for r in results:
                if key_of(Path(r["path"]), root, all_keys) == k and r["stream_hw"]:
                    odd[r["stream_hw"]].append(r["path"])
            major = max(odd, key=lambda d: len(odd[d]))
            for dims, paths in odd.items():
                if dims == major:
                    continue
                eps = sorted(
                    e["episode_index"] for e in episodes
                    if any(str(root / rel) in set(paths) for rel in e["videos"].values())
                )
                print(f"\n     {k} 有 {len(paths)} 个文件是 {dims[0]}x{dims[1]}"
                      f"（主流是 {major[0]}x{major[1]}）")
                if eps:
                    print(f"       涉及 episode: {eps[:10]}{' ...' if len(eps) > 10 else ''}"
                          f"  共 {len(eps)} 个")
                    ranges = [(e["start"], e["end"]) for e in episodes
                              if e["episode_index"] in set(eps)]
                    print(f"       全局帧区间示例: {ranges[:5]}{' ...' if len(ranges) > 5 else ''}")
                else:
                    print(f"       示例文件: {[str(Path(p).relative_to(root)) for p in paths[:3]]}")
        print("\n     单文件一 episode 的 v2.1 布局下仍然能跑（每个文件自身是自洽的），")
        print("     但一旦转成 v3.0（多个 episode 拼进同一个 mp4），拼接后的文件就是")
        print("     变分辨率的，训练必然报上面那个错。同时不同分辨率也无法直接 collate。")
    mismatched = [k for k in keys
                  if k in declared and any(d is not None and d != declared[k] for d in per_key[k])]
    if mismatched:
        if exit_code == 0:
            exit_code = 2
        print(f"\n  ⚠️  这些 key 的实际分辨率与 info.json 声明不一致: {mismatched}")
    if exit_code == 0:
        print("  ✅ 所有视频分辨率自洽，且与 info.json 声明一致。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "root": str(root),
            "declared": {k: list(v) for k, v in declared.items()},
            "per_key": {k: {str(d): c for d, c in v.items()} for k, v in per_key.items()},
            "broken": broken,
            "results": results,
        }, indent=2, ensure_ascii=False))
        print(f"\n报告已写入: {args.json}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
