"""
FF++ c40 비디오에서 프레임을 추출하는 스크립트.
클립당 N장만 균등 추출해서 이미지 수를 고정할 수 있음.

# Deepfakes  Face2Face  NeuralTextures

사용법:
    # 마스크 추출용 (train split 기준, 클립당 20장)
    python extract_frames.py \
        --video_dir "C:/Temp/TP2606_deepfake/FFHQ_pp/manipulated_sequences/Deepfakes/c23/videos" \
        --out_dir "C:/Temp/TP2606_deepfake/frames/Deepfakes" \
        --frames_per_clip 20

    # test split만 추출할 경우 (나중에 평가용)
    python extract_frames.py \
        --video_dir "C:/Temp/TP2606_deepfake/FFHQ_pp/manipulated_sequences/Deepfakes/c23/videos" \
        --out_dir "C:/Temp/TP2606_deepfake/frames_test/Deepfakes" \
        --frames_per_clip 20 \
        --split_json "C:/Temp/TP2606_deepfake/FFHQ_pp/dataset/splits/test.json"
"""

import argparse
import json
import os
import cv2
from pathlib import Path
from tqdm import tqdm


def get_video_ids_from_split(split_json_path):
    """
    test.json / train.json 읽어서 해당하는 비디오 파일명 집합 반환.
    FF++ 파일명 형식: {target}_{source}.mp4
    """
    with open(split_json_path, "r") as f:
        pairs = json.load(f)
    # ["953", "974"] → "953_974"
    return {f"{target}_{source}" for target, source in pairs}


def extract_frames(video_path: Path, out_dir: Path, frames_per_clip: int):
    """비디오 1개에서 균등 간격으로 N장 추출."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        print(f"  [경고] 프레임 수 읽기 실패: {video_path.name}")
        cap.release()
        return 0

    # 균등 간격 프레임 인덱스 선택
    n = min(frames_per_clip, total)
    indices = [int(i * total / n) for i in range(n)]
    indices_set = set(indices)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices_set:
            out_path = out_dir / f"{frame_idx:05d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main(video_dir, out_dir, frames_per_clip, split_json):
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)

    all_videos = sorted(video_dir.glob("*.mp4"))
    if len(all_videos) == 0:
        print(f"[오류] mp4 파일이 없습니다: {video_dir}")
        return

    # split_json이 지정된 경우 해당 클립만 필터링
    if split_json:
        target_ids = get_video_ids_from_split(split_json)
        videos = [v for v in all_videos if v.stem in target_ids]
        print(f"Split 필터링: 전체 {len(all_videos)}개 → {len(videos)}개")
    else:
        videos = all_videos
        print(f"전체 비디오: {len(videos)}개")

    total_frames = 0
    for video_path in tqdm(videos, desc="프레임 추출 중"):
        clip_out = out_dir / video_path.stem  # e.g. frames/Deepfakes/953_974/
        saved = extract_frames(video_path, clip_out, frames_per_clip)
        total_frames += saved

    print(f"\n완료! 총 추출 이미지 수: {total_frames:,}장")
    print(f"저장 위치: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",       required=True,  help="mp4 파일들이 있는 폴더")
    parser.add_argument("--out_dir",         required=True,  help="프레임 저장 폴더")
    parser.add_argument("--frames_per_clip", type=int, default=20, help="클립당 추출할 프레임 수 (기본 20)")
    parser.add_argument("--split_json",      default=None,   help="test.json 등 split 파일 경로 (없으면 전체 추출)")
    args = parser.parse_args()

    main(args.video_dir, args.out_dir, args.frames_per_clip, args.split_json)