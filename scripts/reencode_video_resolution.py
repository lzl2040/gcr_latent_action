#!/usr/bin/env python
"""把 LeRobot 数据集里分辨率不一致的视频重新编码成统一分辨率。

为什么需要
----------
torchcodec 的批量解码 API 会按 **流元数据** 一次性预分配 (N, H, W, 3)，而 mp4 的
元数据只记录第一段的分辨率。所以只要一个 mp4 内部混了多种分辨率，就会抛:

    Expected pre-allocated tensor of shape 480x640x3, got [720, 1280, 3]

v2.1 布局一个 episode 一个 mp4，即使不同 episode 分辨率不同也各自自洽、能跑；
但转成 v3.0 会把多个 episode 拼进同一个 mp4，跨分辨率边界的文件就变成变分辨率的，
训练必炸。所以要在转 v3.0 **之前** 把分辨率统一掉。

先用 scripts/check_video_resolution_consistency.py 确认问题，再用本脚本修。

安全性
------
- 帧数、帧率必须与原文件完全一致（episodes.jsonl 的 length 和时间戳依赖它），
  校验不过就跳过该文件，原文件不动;
- 新文件先写到同目录的临时名，校验通过后才 os.replace 原子替换;
- 原文件默认移动到 --backup-dir（保持相对路径），可 --no-backup 直接丢弃;
- 编码参数默认与 repo 的 encode_video_frames 一致
  (libsvtav1 / yuv420p / crf 30 / g 2)。

用法
----
    conda activate lerobot_v2

    # 先看会改哪些文件（不动数据）
    python scripts/reencode_video_resolution.py \
        --root /Data/lerobot_data_ort6d/v30/RoboMind_full/franka_3rgb --dry-run

    # 真正执行: 统一到 info.json 声明的分辨率
    python scripts/reencode_video_resolution.py \
        --root /Data/lerobot_data_ort6d/v30/RoboMind_full/franka_3rgb --workers 8

    # 反过来: 统一到占多数的那个分辨率，并同步改 info.json
    python scripts/reencode_video_resolution.py --root ... \
        --target majority --update-info

    # 保持长宽比，用黑边补齐（默认是直接拉伸）
    python scripts/reencode_video_resolution.py --root ... --strategy pad
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# 探测
# ============================================================================

def probe(path: str) -> dict | None:
    """读出重编码前后必须比对的那几项。"""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,nb_frames,r_frame_rate,pix_fmt,codec_name",
             "-of", "json", path],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return None
        streams = json.loads(proc.stdout).get("streams") or []
        if not streams:
            return None
        s = streams[0]
        nb = s.get("nb_frames")
        if nb in (None, "N/A"):
            count = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                 "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=600,
            )
            nb = count.stdout.strip() or None
        return {
            "height": int(s["height"]),
            "width": int(s["width"]),
            "nb_frames": int(nb) if nb else None,
            "fps": s.get("r_frame_rate"),
            "pix_fmt": s.get("pix_fmt"),
            "codec": s.get("codec_name"),
        }
    except Exception:
        return None


def decode_check(path: str, n_samples: int = 16) -> str | None:
    """跑一次训练用的批量解码，返回错误字符串（None 表示正常）。"""
    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(path, seek_mode="approximate")
        total = int(decoder.metadata.num_frames or 0)
        if total <= 0:
            return "num_frames <= 0"
        step = max(1, total // max(1, n_samples))
        indices = sorted({min(i, total - 1) for i in range(0, total, step)})[:n_samples]
        decoder.get_frames_at(indices=indices or [0])
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


# ============================================================================
# 重编码
# ============================================================================

def build_filter(target_h: int, target_w: int, strategy: str) -> str:
    if strategy == "stretch":
        return f"scale={target_w}:{target_h}:flags=bicubic"
    if strategy == "pad":
        return (f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=bicubic,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black")
    if strategy == "crop":
        return (f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=bicubic,"
                f"crop={target_w}:{target_h}")
    raise ValueError(f"未知 strategy: {strategy}")


def reencode_one(job: dict) -> dict:
    """重编码单个文件并逐项校验。任何一步不过就还原，绝不动原文件。"""
    src = Path(job["path"])
    out = {"path": str(src), "ok": False, "error": None,
           "before": job["probe"], "after": None}
    tmp = src.with_name(f".{src.name}.reencode.{os.getpid()}.mp4")
    try:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
               "-vf", build_filter(job["target_h"], job["target_w"], job["strategy"]),
               "-c:v", job["vcodec"], "-pix_fmt", job["pix_fmt"], "-an"]
        if job["crf"] is not None:
            cmd += ["-crf", str(job["crf"])]
        if job["g"] is not None:
            cmd += ["-g", str(job["g"])]
        if job["fast_decode"]:
            cmd += ["-svtav1-params", f"fast-decode={job['fast_decode']}"]
        cmd.append(str(tmp))

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            out["error"] = f"ffmpeg 失败: {proc.stderr.strip()[:300]}"
            return out

        after = probe(str(tmp))
        out["after"] = after
        if after is None:
            out["error"] = "重编码后的文件无法探测"
            return out

        before = job["probe"]
        if (after["height"], after["width"]) != (job["target_h"], job["target_w"]):
            out["error"] = (f"分辨率不对: 期望 {job['target_h']}x{job['target_w']}, "
                            f"得到 {after['height']}x{after['width']}")
            return out
        if before.get("nb_frames") and after.get("nb_frames") != before["nb_frames"]:
            out["error"] = (f"帧数变了: {before['nb_frames']} -> {after['nb_frames']}"
                            "（会破坏 episodes.jsonl 的 length / 时间戳）")
            return out
        if before.get("fps") and after.get("fps") != before["fps"]:
            out["error"] = f"帧率变了: {before['fps']} -> {after['fps']}"
            return out
        err = decode_check(str(tmp))
        if err:
            out["error"] = f"重编码后仍解码失败: {err}"
            return out

        # 校验全过，先备份原文件再原子替换
        if job["backup_dir"]:
            backup = Path(job["backup_dir"]) / job["rel"]
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(backup))
        os.replace(tmp, src)
        out["ok"] = True
        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ============================================================================
# 主流程
# ============================================================================

def key_of(path: Path, root: Path, keys: list[str]) -> str:
    parts = set(path.relative_to(root).parts)
    for k in keys:
        if k in parts:
            return k
    return "<unknown>"


def parse_target(spec: str) -> tuple[int, int] | None:
    if spec in ("declared", "majority"):
        return None
    sep = "x" if "x" in spec.lower() else ":"
    h, w = spec.lower().split(sep)
    return int(h), int(w)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把数据集里分辨率不一致的视频重新编码成统一分辨率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", type=Path,
                        default=Path("/Data/lerobot_data_ort6d/v30/RoboMind_full/franka_3rgb"))
    parser.add_argument("--keys", nargs="*", default=None, help="只处理这些 video key")
    parser.add_argument("--target", default="declared",
                        help="目标分辨率: declared(按 info.json, 默认) / majority(按多数) / HxW")
    parser.add_argument("--strategy", default="stretch", choices=["stretch", "pad", "crop"],
                        help="stretch=直接拉伸(默认, 与下游 resize 到正方形的行为一致); "
                             "pad=保长宽比加黑边; crop=保长宽比裁切")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--pix-fmt", default="yuv420p")
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--g", type=int, default=2, help="关键帧间隔（repo 默认 2，利于随机访问）")
    parser.add_argument("--fast-decode", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="只列出会改哪些文件")
    parser.add_argument("--backup-dir", type=Path, default=None,
                        help="原文件移动到这里（默认 <root>/.reencode_backup）")
    parser.add_argument("--no-backup", action="store_true", help="不备份，直接覆盖原文件")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="连已经是目标分辨率的也重编码")
    parser.add_argument("--update-info", action="store_true",
                        help="重编码后把 info.json 的 shape / video.height / video.width 改成目标值")
    args = parser.parse_args()

    root = args.root.resolve()
    info = json.loads((root / "meta" / "info.json").read_text())
    all_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    keys = args.keys or all_keys
    declared = {}
    for k, v in info.get("features", {}).items():
        if v.get("dtype") == "video" and v.get("shape"):
            declared[k] = (int(v["shape"][0]), int(v["shape"][1]))

    print("=" * 78)
    print(f"数据集: {root}")
    print(f"codebase_version: {info.get('codebase_version')}")
    print(f"目标: {args.target}   策略: {args.strategy}")
    print(f"编码: {args.vcodec} / {args.pix_fmt} / crf={args.crf} / g={args.g}")
    print("=" * 78)

    videos = sorted(p for p in (root / "videos").rglob("*.mp4")
                    if key_of(p, root, all_keys) in keys and not p.name.startswith("."))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print("!! 没找到视频文件")
        return 1

    print(f"\n探测 {len(videos)} 个视频...")
    probes: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(args.workers, 8)) as pool:
        futures = {pool.submit(probe, str(p)): str(p) for p in videos}
        for i, fut in enumerate(as_completed(futures), 1):
            probes[futures[fut]] = fut.result()
            if i % 2000 == 0 or i == len(videos):
                print(f"   ... {i}/{len(videos)}")

    # ---- 决定每个 key 的目标分辨率 ----
    per_key = defaultdict(Counter)
    for path, info_p in probes.items():
        if info_p:
            per_key[key_of(Path(path), root, all_keys)][(info_p["height"], info_p["width"])] += 1

    explicit = parse_target(args.target)
    targets: dict[str, tuple[int, int]] = {}
    print("\n各 key 现状与目标:")
    for k in sorted(per_key):
        if explicit:
            tgt = explicit
        elif args.target == "majority":
            tgt = per_key[k].most_common(1)[0][0]
        else:
            if k not in declared:
                print(f"  {k}: info.json 没有 shape，跳过")
                continue
            tgt = declared[k]
        targets[k] = tgt
        dist = ", ".join(f"{h}x{w}×{c}" for (h, w), c in per_key[k].most_common())
        note = ""
        if k in declared and tgt != declared[k]:
            note = f"  (与 info.json 声明 {declared[k][0]}x{declared[k][1]} 不同)"
        print(f"  {k}\n      现状: {dist}\n      目标: {tgt[0]}x{tgt[1]}{note}")

    # ---- 挑出要改的文件 ----
    jobs = []
    unreadable = []
    for path, info_p in probes.items():
        k = key_of(Path(path), root, all_keys)
        if k not in targets:
            continue
        if info_p is None:
            unreadable.append(path)
            continue
        tgt = targets[k]
        if not args.force and (info_p["height"], info_p["width"]) == tgt:
            continue
        jobs.append({
            "path": path,
            "rel": str(Path(path).relative_to(root)),
            "probe": info_p,
            "target_h": tgt[0], "target_w": tgt[1],
            "strategy": args.strategy, "vcodec": args.vcodec,
            "pix_fmt": args.pix_fmt, "crf": args.crf, "g": args.g,
            "fast_decode": args.fast_decode,
            "backup_dir": None,
        })
    jobs.sort(key=lambda j: j["rel"])

    if unreadable:
        print(f"\n⚠️  {len(unreadable)} 个文件 ffprobe 读不了，已跳过:")
        for p in unreadable[:5]:
            print(f"     {Path(p).relative_to(root)}")

    print(f"\n需要重编码: {len(jobs)} / {len(videos)}")
    if not jobs:
        print("✅ 所有视频已经是目标分辨率，无事可做。")
        return 0

    by_key = Counter(key_of(Path(j["path"]), root, all_keys) for j in jobs)
    for k, c in by_key.most_common():
        print(f"   {k}: {c} 个")
    print("\n示例:")
    for j in jobs[:5]:
        b = j["probe"]
        print(f"   {j['rel']}  {b['height']}x{b['width']} -> {j['target_h']}x{j['target_w']}"
              f"  ({b['nb_frames']} 帧)")

    if args.dry_run:
        print("\n--dry-run: 不做任何修改。")
        return 0

    backup_dir = None
    if not args.no_backup:
        backup_dir = (args.backup_dir or root / ".reencode_backup").resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n原文件将备份到: {backup_dir}")
    else:
        print("\n⚠️  --no-backup: 原文件会被直接覆盖")
    for j in jobs:
        j["backup_dir"] = str(backup_dir) if backup_dir else None

    print(f"\n开始重编码（{args.workers} 并行）...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(reencode_one, j): j["rel"] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 50 == 0 or i == len(jobs):
                done = sum(1 for r in results if r["ok"])
                print(f"   ... {i}/{len(jobs)}  成功 {done}")

    failed = [r for r in results if not r["ok"]]
    print("\n" + "=" * 78)
    print("【结果】")
    print("=" * 78)
    print(f"  成功: {len(results) - len(failed)} / {len(results)}")
    if failed:
        print(f"  失败: {len(failed)}（原文件未被修改）")
        for r in failed[:20]:
            print(f"    ❌ {Path(r['path']).relative_to(root)}\n       {r['error']}")

    if args.update_info and not failed:
        changed = False
        for k, tgt in targets.items():
            feat = info["features"].get(k)
            if not feat:
                continue
            if feat.get("shape") and tuple(feat["shape"][:2]) != tgt:
                feat["shape"] = [tgt[0], tgt[1], feat["shape"][2] if len(feat["shape"]) > 2 else 3]
                changed = True
            vinfo = feat.get("info") or {}
            if "video.height" in vinfo or "video.width" in vinfo:
                vinfo["video.height"], vinfo["video.width"] = tgt[0], tgt[1]
                feat["info"] = vinfo
                changed = True
        if changed:
            path = root / "meta" / "info.json"
            shutil.copy2(path, path.with_suffix(".json.bak"))
            path.write_text(json.dumps(info, indent=4))
            print(f"\n  info.json 已更新（原文件备份为 {path.name}.bak）")

    if backup_dir:
        print(f"\n  原文件在 {backup_dir}，确认无误后可删除。")
    print("\n  建议复查:")
    print(f"    python scripts/check_video_resolution_consistency.py --root {root}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
