#!/usr/bin/env python
"""
Script to check for corrupted video files in a LeRobot dataset.

This script scans all video files in a dataset and reports any that fail to decode.

Usage:
    conda activate lerobot_v2
    python scripts/check_corrupted_videos.py --root /Data/lerobot_data_ort6d/v30/fractal20220817_data_lerobot
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_video_file(video_path: str) -> dict:
    """Check if a video file is valid and can be decoded.
    
    Returns:
        dict with keys: 'path', 'valid', 'error', 'codec', 'duration'
    """
    result = {
        'path': video_path,
        'valid': True,
        'error': None,
        'codec': None,
        'duration': None,
    }
    
    # First check if file exists
    if not os.path.exists(video_path):
        result['valid'] = False
        result['error'] = "File not found"
        return result
    
    # Use ffprobe to check video metadata
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,duration",
            "-of", "json",
            video_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if proc.returncode != 0:
            result['valid'] = False
            result['error'] = f"ffprobe failed: {proc.stderr}"
            return result
        
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        if not streams:
            result['valid'] = False
            result['error'] = "No video stream found"
            return result
        
        stream = streams[0]
        result['codec'] = stream.get("codec_name")
        result['duration'] = stream.get("duration")
        
    except subprocess.TimeoutExpired:
        result['valid'] = False
        result['error'] = "ffprobe timeout"
        return result
    except Exception as e:
        result['valid'] = False
        result['error'] = f"ffprobe error: {str(e)}"
        return result
    
    # Try to decode a frame using torchcodec
    try:
        from torchcodec.decoders import VideoDecoder
        
        decoder = VideoDecoder(video_path, seek_mode="approximate")
        metadata = decoder.metadata
        
        # Try to get the first frame
        if metadata.num_frames > 0:
            frame = decoder.get_frame_at(index=0)
        
        # Close decoder
        if hasattr(decoder, 'close'):
            decoder.close()
            
    except Exception as e:
        result['valid'] = False
        result['error'] = f"Decode error: {str(e)}"
        return result
    
    return result


def scan_dataset_videos(root: str, video_keys: list[str] = None):
    """Scan all video files in a dataset and check for corruption.
    
    Args:
        root: Root directory of the dataset
        video_keys: List of video keys to check (e.g., ['observation.images.primary'])
                   If None, will auto-discover from videos/ directory
    """
    root = Path(root)
    videos_dir = root / "videos"
    
    if not videos_dir.exists():
        print(f"No videos directory found at: {videos_dir}")
        return
    
    # Auto-discover video keys if not provided
    if video_keys is None:
        video_keys = []
        for item in videos_dir.iterdir():
            if item.is_dir():
                video_keys.append(item.name)
    
    if not video_keys:
        print("No video keys found in videos directory")
        return
    
    print(f"Scanning video keys: {video_keys}")
    
    # Collect all video files
    video_files = []
    for vid_key in video_keys:
        vid_dir = videos_dir / vid_key
        if vid_dir.exists():
            for mp4_file in vid_dir.rglob("*.mp4"):
                video_files.append(str(mp4_file))
    
    print(f"Found {len(video_files)} video files to check")
    
    # Check videos in parallel
    corrupted_files = []
    valid_count = 0
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(check_video_file, vf): vf for vf in video_files}
        
        for future in as_completed(futures):
            result = future.result()
            if result['valid']:
                valid_count += 1
            else:
                corrupted_files.append(result)
                print(f"[CORRUPTED] {result['path']}")
                print(f"  Error: {result['error']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total video files: {len(video_files)}")
    print(f"Valid files: {valid_count}")
    print(f"Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print("\n" + "=" * 60)
        print("CORRUPTED FILES LIST")
        print("=" * 60)
        for cf in corrupted_files:
            print(f"\n{cf['path']}")
            print(f"  Codec: {cf['codec']}")
            print(f"  Error: {cf['error']}")
        
        # Save corrupted files list to a file
        output_file = root / "corrupted_videos_report.json"
        with open(output_file, 'w') as f:
            json.dump(corrupted_files, f, indent=2)
        print(f"\nReport saved to: {output_file}")
    
    return corrupted_files


def main():
    parser = argparse.ArgumentParser(description="Check for corrupted video files in a LeRobot dataset")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--video_keys", type=str, nargs='*', default=None, 
                        help="List of video keys to check (e.g., observation.images.primary)")
    
    args = parser.parse_args()
    
    print(f"Checking videos in: {args.root}")
    scan_dataset_videos(args.root, args.video_keys)


if __name__ == "__main__":
    main()