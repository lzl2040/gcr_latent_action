#!/usr/bin/env python3
"""
用 azcopy 把 Azure Blob 数据下载到本地。

参考 lola_lerobot 的 download_azure_azcopy.py 精简而来，主要差异:
  - 参考脚本在 Azure 云主机上用 MSI (托管身份) 登录;
    本地服务器没有 MSI, 本脚本支持 SAS token 认证（本地常用），同时保留 MSI 模式
  - 去掉 GPU 保活 / blobfuse 修补等训练平台相关逻辑，只保留下载 + 重试 + 断点续传

认证方式（二选一）:
  1. SAS token（本地服务器）: --sas "<token>" 或环境变量 AZURE_BLOB_SAS_TOKEN
     （token 是 "?sv=..." 或 "sv=..." 形式的查询串，会自动拼到 URL 上）
  2. MSI（Azure 云主机）: --msi，脚本会 azcopy login --identity

用法:
    # 下载整个目录（blob 前缀）到本地
    python download_blob_azcopy.py \
        --account <存储账号> --container <容器名> \
        --src "ftp-subset/Unit_Bimanual" \
        --dst /data_16T/deepseek/lzl/data/ftp-subset/Unit_Bimanual \
        --sas "$AZURE_BLOB_SAS_TOKEN"

    # 也可以直接给完整 URL
    python download_blob_azcopy.py \
        --src "https://<账号>.blob.core.windows.net/<容器>/ftp-subset/Unit_Bimanual" \
        --dst /data_16T/deepseek/lzl/data/ftp-subset/Unit_Bimanual \
        --sas "$AZURE_BLOB_SAS_TOKEN"

    # 云主机 MSI 模式
    python download_blob_azcopy.py --msi --account ... --container ... --src ... --dst ...

    # 上传模式: 把本地目录传回 blob（如集群转换完 lerobot 数据集后回传）
    python download_blob_azcopy.py --upload --msi \
        --src /scratch/lerobot_out/VisuoTactile_QINGLOONG \
        --dst "https://<账号>.blob.core.windows.net/<容器>/lerobot_ftp/VisuoTactile_QINGLOONG"
    # 注意: 上传时 SAS token 需要写权限（sp 含 cw）；azcopy 同样会在
    # 目标 URL 父路径下重建本地叶子目录名

azcopy 二进制: 默认自动下载到脚本同目录的 ./azcopy（已有则复用），
可用 --azcopy-bin 指定已有二进制路径。
"""

import argparse
import logging
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AZCOPY_DOWNLOAD_URL = "https://aka.ms/downloadazcopy-v10-linux"
MAX_RETRIES = 5
RETRY_DELAY_S = 10


# ============================================================================
# 后台监控: 磁盘剩余空间 + 容器内存用量
# 集群上 exit 137 (SIGKILL) 常见于 OOMKill: 写盘时 page cache 计入容器内存
# 配额, 或 azcopy 自身内存膨胀。下载期间周期性记录, 被杀前可从日志定位。
# ============================================================================

def _read_cgroup_memory() -> str:
    """读取容器内存用量/上限（兼容 cgroup v1 / v2），返回描述字符串。"""
    # cgroup v2
    v2_cur = Path("/sys/fs/cgroup/memory.current")
    v2_max = Path("/sys/fs/cgroup/memory.max")
    if v2_cur.exists():
        try:
            cur = int(v2_cur.read_text().strip())
            raw = v2_max.read_text().strip() if v2_max.exists() else "max"
            limit = "max(无限制)" if raw == "max" else f"{int(raw) / 2**30:.1f}GiB"
            return f"cgroup(v2) 内存 {cur / 2**30:.2f}GiB / {limit}"
        except Exception:
            pass
    # cgroup v1
    v1_cur = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    v1_max = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1_cur.exists():
        try:
            cur = int(v1_cur.read_text().strip())
            limit = int(v1_max.read_text().strip()) if v1_max.exists() else 0
            # v1 无限制时 limit 是一个接近 int64 上限的巨大值
            limit_s = f"{limit / 2**30:.1f}GiB" if 0 < limit < 1 << 60 else "max(无限制)"
            return f"cgroup(v1) 内存 {cur / 2**30:.2f}GiB / {limit_s}"
        except Exception:
            pass
    return "cgroup 内存信息不可用"


def _read_self_rss_gb() -> float:
    """当前进程 RSS（GiB）。"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return -1.0


def _read_pid_rss_gb(pid: int) -> float:
    """指定进程 RSS（GiB），进程不存在返回 -1。"""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return -1.0


def _read_dirty_gb() -> str:
    """读 /proc/meminfo 的 Dirty + Writeback（GiB）。
    Dirty 高说明写盘数据在 page cache 里排队等回写——cgroup v2 把这些算进
    内存配额，是下载大文件时 OOMKill 的常见元凶。"""
    dirty = writeback = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("Dirty:"):
                    dirty = int(line.split()[1]) / 2**20
                elif line.startswith("Writeback:"):
                    writeback = int(line.split()[1]) / 2**20
    except Exception:
        pass
    if dirty is None:
        return "Dirty 不可用"
    return f"Dirty {dirty:.2f}GiB, Writeback {writeback:.2f}GiB"


# 当前正在运行的 azcopy 子进程 PID（由 run_azcopy_download 设置），
# 供监控线程读取 azcopy 自身内存
_current_azcopy_pid = None


def monitor_resources(paths: list, interval: float, stop_event: threading.Event) -> None:
    """后台线程: 每隔 interval 秒记录各路径剩余空间 + 容器内存用量。"""
    while not stop_event.is_set():
        parts = []
        for p in paths:
            try:
                st = os.statvfs(p)
                free = st.f_bavail * st.f_frsize / 2**30
                total = st.f_blocks * st.f_frsize / 2**30
                parts.append(f"{p}: 剩余 {free:.1f}GiB / 共 {total:.1f}GiB")
            except Exception as e:
                parts.append(f"{p}: statvfs 失败 ({e})")
        parts.append(_read_cgroup_memory())
        parts.append(_read_dirty_gb())
        rss = _read_self_rss_gb()
        if rss >= 0:
            parts.append(f"下载进程 RSS {rss:.2f}GiB")
        if _current_azcopy_pid is not None:
            az_rss = _read_pid_rss_gb(_current_azcopy_pid)
            if az_rss >= 0:
                parts.append(f"azcopy RSS {az_rss:.2f}GiB")
        logger.info(f"[监控] " + " | ".join(parts))
        stop_event.wait(interval)


# ============================================================================
# azcopy 二进制: 自动下载 / 复用
# ============================================================================

def ensure_azcopy(azcopy_bin: Path) -> Path:
    """确保 azcopy 可用: 已在 PATH 或指定路径则复用，否则自动下载解压。"""
    # 1. 指定路径已存在
    if azcopy_bin.exists() and os.access(azcopy_bin, os.X_OK):
        logger.info(f"使用已有 azcopy: {azcopy_bin}")
        return azcopy_bin

    # 2. PATH 里有
    from shutil import which
    in_path = which("azcopy")
    if in_path:
        logger.info(f"使用 PATH 中的 azcopy: {in_path}")
        return Path(in_path)

    # 3. 自动下载（官方 tar.gz，内含 azcopy_linux_amd64_*/azcopy）
    logger.info(f"azcopy 不存在，从 {AZCOPY_DOWNLOAD_URL} 下载...")
    azcopy_bin.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(AZCOPY_DOWNLOAD_URL, tmp_path)
        with tarfile.open(tmp_path, "r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("/azcopy"))
            member.name = "azcopy"  # 拍平目录结构
            tf.extract(member, azcopy_bin.parent, filter="data")
        extracted = azcopy_bin.parent / "azcopy"
        extracted.chmod(extracted.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if extracted != azcopy_bin:
            extracted.rename(azcopy_bin)
        logger.info(f"azcopy 已安装到: {azcopy_bin}")
        return azcopy_bin
    finally:
        tmp_path.unlink(missing_ok=True)


# ============================================================================
# blob 引用解析（参考 resolve_blob_ref）
# ============================================================================

def resolve_blob_url(src: str, account: str = None, container: str = None) -> str:
    """
    将源引用解析为完整 blob URL:
      - 完整 https:// URL → 原样返回
      - 否则需要 --account/--container，拼为
        https://{account}.blob.core.windows.net/{container}/{src}
        （src 开头多余的 "/" 会被去掉）
    """
    if src.startswith("https://") or src.startswith("http://"):
        return src
    if not account or not container:
        raise ValueError("--src 不是完整 URL 时必须提供 --account 和 --container")
    return f"https://{account}.blob.core.windows.net/{container}/{src.lstrip('/')}"


def append_sas(url: str, sas: str) -> str:
    """把 SAS token 拼到 URL 上（token 可带或不带前导 '?'）。"""
    if not sas:
        return url
    sep = "&" if "?" in url else "?"
    return url + sep + sas.lstrip("?")


# ============================================================================
# azcopy 调用: 登录 / 下载 / 重试 / 续传
# ============================================================================

def azcopy_login_msi(azcopy: Path) -> None:
    """MSI 模式: azcopy login --identity（仅 Azure 云主机可用）。"""
    env = {**os.environ, "AZCOPY_AUTO_LOGIN_TYPE": "MSI"}
    subprocess.run([str(azcopy), "login", "--identity"], check=True, env=env)
    logger.info("azcopy MSI 登录成功")


def run_azcopy_copy(azcopy: Path, src: str, target: str, overwrite: str = "ifSourceNewer",
                    cap_mbps: float = None, extra_args: list = None,
                    use_msi: bool = False) -> None:
    """
    执行 azcopy copy，带重试与断点续传（参考 download_azure_azcopy.py 的重试逻辑）:
      - 首次: azcopy copy <src> <target> --recursive=true
      - 失败重试时优先 azcopy jobs resume <jobId>，resume 失败则重新 copy
        （--overwrite=ifSourceNewer 保证已传输的文件不会重传）

    src / target 为 azcopy 的最终参数（下载: blob URL → 本地目录;
    上传: 本地目录 → blob URL），调用方负责按 azcopy 目录语义换算好。
    """
    env = dict(os.environ)
    if use_msi:
        env["AZCOPY_AUTO_LOGIN_TYPE"] = "MSI"

    cmd_base = [
        str(azcopy), "copy", src, target,
        "--recursive=true", f"--overwrite={overwrite}",
    ]
    if cap_mbps:
        cmd_base.append(f"--cap-mbps={cap_mbps}")
    cmd_base += (extra_args or [])

    job_id = None
    for attempt in range(1, MAX_RETRIES + 1):
        if job_id:
            cmd = [str(azcopy), "jobs", "resume", job_id]
            logger.info(f"第 {attempt}/{MAX_RETRIES} 次尝试: 续传 job {job_id}")
        else:
            cmd = cmd_base
            cap_info = f", 限速 {cap_mbps} Mb/s" if cap_mbps else ""
            logger.info(f"第 {attempt}/{MAX_RETRIES} 次尝试: {src} → {target}{cap_info}")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)
        global _current_azcopy_pid
        _current_azcopy_pid = proc.pid  # 供监控线程读取 azcopy 自身内存
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            # 只打关键行: 进度 / 吞吐 / 失败 / JobId，避免刷屏
            if re.search(r"Throughput|Failed|Job.*Id|Final Job Status|Total Number of Bytes", line):
                logger.info(f"  [azcopy] {line}")
            m = re.search(r"Job ([0-9a-f-]{36}) has started", line)
            if m:
                job_id = m.group(1)
        proc.wait()
        _current_azcopy_pid = None

        if proc.returncode == 0:
            logger.info("✅ 传输完成")
            return

        logger.warning(f"azcopy 退出码 {proc.returncode}")
        if attempt < MAX_RETRIES:
            if job_id is None:
                logger.warning("未拿到 JobId，下次将重新 copy（已传输文件会跳过）")
            logger.warning(f"{RETRY_DELAY_S} 秒后重试...")
            time.sleep(RETRY_DELAY_S)
            # resume 失败过一次就放弃 job_id，回退到 copy --overwrite=ifSourceNewer
            if attempt >= 2:
                job_id = None

    raise RuntimeError(f"传输失败，已重试 {MAX_RETRIES} 次: {src} → {target}")


def run_azcopy_download(azcopy: Path, url: str, dst: Path, overwrite: str = "ifSourceNewer",
                        cap_mbps: float = None, extra_args: list = None,
                        use_msi: bool = False) -> None:
    """
    下载。注意 azcopy v10 目录语义: 拷贝目录时会在目标下重建叶子目录名，
    例如 copy ".../Unit_Bimanual" /dst_parent → /dst_parent/Unit_Bimanual/...
    所以这里取 dst 的父目录作为 azcopy 目标。
    """
    dst = dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_azcopy_copy(azcopy, url, str(dst.parent), overwrite, cap_mbps, extra_args, use_msi)


def run_azcopy_upload(azcopy: Path, local_dir: Path, url: str, overwrite: str = "ifSourceNewer",
                      cap_mbps: float = None, extra_args: list = None,
                      use_msi: bool = False) -> None:
    """
    上传。上传与下载的目录语义对称: azcopy 会在目标 URL 下重建本地叶子目录名，
    例如 copy /data/Unit_Bimanual ".../container/ftp-subset" →
    .../container/ftp-subset/Unit_Bimanual/...
    所以这里取 url 的父路径作为 azcopy 目标。
    """
    local_dir = local_dir.resolve()
    if not local_dir.is_dir():
        raise ValueError(f"上传源必须是已存在的本地目录: {local_dir}")
    run_azcopy_copy(azcopy, str(local_dir), _url_parent(url),
                    overwrite, cap_mbps, extra_args, use_msi)


def _url_parent(url: str) -> str:
    """取 blob URL 的父路径（保留 query string，如 SAS token）。"""
    no_query, sep, query = url.partition("?")
    parent = no_query.rstrip("/").rsplit("/", 1)[0]
    return parent + sep + query

# FTP_SUBSET_ROOT = Path("/mnt/in_agibot/robot_dataset/ftp-1-unzip/FTP-1-Dataset")
# OUTPUT_ROOT = Path("/mnt/in_agibot/robot_dataset/lerobot-format-v30/FTP-1")
# ============================================================================
# 路径配置
# ============================================================================
FTP_SUBSET_ROOT = Path("/scratch/ftp-1")
OUTPUT_ROOT = Path("/scratch/lerobot_ftp")

def main():
    parser = argparse.ArgumentParser(
        description="用 azcopy 把 Azure Blob 数据下载到本地（支持 SAS / MSI 认证，断点续传）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--src", required=True,
                        help="源: 完整 blob URL，或容器内路径（需配合 --account/--container）。"
                             "--upload 模式下为本地目录")
    parser.add_argument("--dst", required=True,
                        help="本地目标目录（azcopy 会在其父目录下重建源叶子目录名）。"
                             "--upload 模式下为 blob URL / 容器内路径")
    parser.add_argument("--upload", action="store_true",
                        help="上传模式: --src 为本地目录, --dst 为 blob 目标。"
                             "SAS token 需要写权限（sp 含 cw）")
    parser.add_argument("--account", default=None, help="Azure 存储账号名（src 非完整 URL 时必填）")
    parser.add_argument("--container", default=None, help="Blob 容器名（src 非完整 URL 时必填）")
    parser.add_argument("--sas", default=os.environ.get("AZURE_BLOB_SAS_TOKEN", ""),
                        help="SAS token（也可用环境变量 AZURE_BLOB_SAS_TOKEN）")
    parser.add_argument("--msi", action="store_true",
                        help="使用 MSI 托管身份登录（仅 Azure 云主机）")
    parser.add_argument("--azcopy-bin", type=Path,
                        default=Path(__file__).resolve().parent / "azcopy",
                        help="azcopy 二进制路径（默认: 脚本同目录 ./azcopy，不存在则自动下载）")
    parser.add_argument("--overwrite", default="ifSourceNewer",
                        choices=["true", "false", "ifSourceNewer"],
                        help="覆盖策略（默认 ifSourceNewer，重复执行可增量续传）")
    parser.add_argument("--cap-mbps", type=float, default=None,
                        help="限速（Mb/s），传给 azcopy 的 --cap-mbps。内存受限的 pod 里"
                             "建议限速，让内核来得及回写 page cache，避免触发 OOMKill")
    parser.add_argument("--extra-args", nargs="*", default=None,
                        help="追加给 azcopy copy 的额外参数（注意: 以 -- 开头的参数"
                             "argparse 无法透传，请用 --extra-args=\"--include-pattern=*.json\" 这种写法）")
    parser.add_argument("--monitor-path", action="append", default=None,
                        help="下载期间周期性记录该路径的剩余空间（可重复传入多个路径），"
                             "默认监控 --dst 所在路径；设为空字符串可关闭")
    parser.add_argument("--monitor-interval", type=float, default=10.0,
                        help="资源监控间隔秒数（默认 10）")
    args = parser.parse_args()

    if not args.msi and not args.sas:
        logger.warning("未提供 SAS token 也未启用 --msi，若容器非公开访问将会 403")

    azcopy = ensure_azcopy(args.azcopy_bin)
    if args.msi:
        azcopy_login_msi(azcopy)

    # ---- 解析 blob 端 URL 并附加认证 ----
    blob_ref = args.dst if args.upload else args.src
    url = resolve_blob_url(blob_ref, args.account, args.container)
    if not args.msi:
        url = append_sas(url, args.sas)

    if args.upload:
        local_src = Path(args.src)
        if not local_src.is_dir():
            parser.error(f"--upload 模式下 --src 必须是已存在的本地目录: {local_src}")
        # 防呆: azcopy 会在目标 URL 父路径下重建本地叶子目录名
        dst_leaf = blob_ref.split("?", 1)[0].rstrip("/").split("/")[-1]
        if dst_leaf != local_src.name:
            logger.warning(
                f"--dst URL 的最后一段 '{dst_leaf}' 与本地目录名 '{local_src.name}' 不一致: "
                f"文件将上传到 {_url_parent(url)}/{local_src.name} "
                f"（若这正是预期位置请忽略）"
            )
        logger.info(f"上传: {local_src} → {url}（azcopy 实际写入其父路径，自动带上叶子目录名）")
        # 上传默认监控本地源所在磁盘
        default_monitor = str(local_src.resolve().parent)
    else:
        # 防呆: azcopy 会在 dst 的父目录下重建源叶子目录名,
        # 若 dst 叶子名与源不一致, 最终落点可能不是用户预期的位置
        dst_path = Path(args.dst)
        src_leaf = args.src.split("?", 1)[0].rstrip("/").split("/")[-1]
        if dst_path.name != src_leaf:
            logger.warning(
                f"--dst 的最后一段 '{dst_path.name}' 与源目录名 '{src_leaf}' 不一致: "
                f"文件将下载到 {dst_path.resolve().parent / src_leaf} "
                f"（若这正是预期位置请忽略）"
            )
        logger.info(f"下载: {args.src}")
        logger.info(f"目标: {dst_path}（azcopy 实际写入其父目录，自动带上叶子目录名）")
        default_monitor = str(dst_path.resolve().parent)

    # 启动后台资源监控（默认监控下载目标/上传源所在路径）
    monitor_paths = args.monitor_path if args.monitor_path is not None else [default_monitor]
    monitor_paths = [p for p in monitor_paths if p]
    stop_monitor = threading.Event()
    monitor_thread = None
    if monitor_paths:
        monitor_thread = threading.Thread(
            target=monitor_resources,
            args=(monitor_paths, args.monitor_interval, stop_monitor),
            daemon=True,
        )
        monitor_thread.start()
        logger.info(f"资源监控已启动: {monitor_paths}, 间隔 {args.monitor_interval}s")

    t0 = time.perf_counter()
    try:
        if args.upload:
            run_azcopy_upload(azcopy, Path(args.src), url, args.overwrite,
                              args.cap_mbps, args.extra_args, args.msi)
        else:
            run_azcopy_download(azcopy, url, Path(args.dst), args.overwrite,
                                args.cap_mbps, args.extra_args, args.msi)
    finally:
        stop_monitor.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)
    logger.info(f"总耗时: {time.perf_counter() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
