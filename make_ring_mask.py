"""
ring [0,1,2] (저주파, 22.0%) 마스크를 직접 생성.
extract_freq_mask.py 의 ring 분할 로직과 동일 (k=8, 256x256).
Xception 재실행 없이 바로 마스크 파일만 재생성.

사용법:
    python make_ring_mask.py --out_path "C:/Temp/TP2606_deepfake/masks/mask_B_lowfreq012.npy" --rings 0 1 2
"""

import argparse
import numpy as np
import cv2


def cluster_spectrum(fft_size=256, k=8):
    H, W = fft_size, fft_size
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max()
    edges = np.linspace(0, r_max, k + 1)

    labels = np.zeros((H, W), dtype=np.int64)
    for i in range(k):
        if i == k - 1:
            mask = (r >= edges[i]) & (r <= edges[i + 1])
        else:
            mask = (r >= edges[i]) & (r < edges[i + 1])
        labels[mask] = i
    return labels


def main(out_path, rings, fft_size=256, k=8):
    labels = cluster_spectrum(fft_size, k)

    B = np.zeros((fft_size, fft_size), dtype=np.float32)
    for c in rings:
        B[labels == c] = 1.0

    np.save(out_path, B)
    print(f"[✓] 저장: {out_path}")
    print(f"    center value B[{fft_size//2},{fft_size//2}] = {B[fft_size//2, fft_size//2]}")
    print(f"    coverage = {B.mean()*100:.1f}%")

    # 시각화
    vis_path = out_path.replace(".npy", "_vis.png")
    mask_vis = (B * 255).astype(np.uint8)
    cv2.imwrite(vis_path, mask_vis)
    print(f"    시각화: {vis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--rings", type=int, nargs="+", required=True, help="포함할 ring 번호들 (예: 0 1 2)")
    parser.add_argument("--fft_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    main(args.out_path, args.rings, args.fft_size, args.k)