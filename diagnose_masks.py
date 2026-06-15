"""
마스크 생성 과정 진단 스크립트.
- 각 위조 유형의 진폭 스펙트럼이 실제로 얼마나 다른지
- K-means labels가 위조 유형별로 얼마나 다른지/같은지
- amp 값의 분포 (정규화 전/후)
를 확인.

사용법:
    python diagnose_masks.py \
        --frames_root "C:/Temp/TP2606_deepfake/frames" \
        --out_dir "C:/Temp/TP2606_deepfake/masks/diag"
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def mean_amplitude_spectrum(frames_dir: Path, fft_size=256, max_frames=2000):
    paths = []
    for clip_dir in sorted(frames_dir.iterdir()):
        if clip_dir.is_dir():
            for png in sorted(clip_dir.glob("*.png")):
                paths.append(png)

    if len(paths) > max_frames:
        step = len(paths) / max_frames
        paths = [paths[int(i * step)] for i in range(max_frames)]

    amp_sum = np.zeros((fft_size, fft_size), dtype=np.float64)
    n = 0
    for p in tqdm(paths, desc=f"  {frames_dir.name}"):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.resize(gray, (fft_size, fft_size)).astype(np.float32)
        fft = np.fft.fftshift(np.fft.fft2(gray_r))
        amp_sum += np.abs(fft)
        n += 1

    return amp_sum / n


def cluster_spectrum(amp_mean, k=8):
    H, W = amp_mean.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).flatten()
    theta = np.arctan2(yy.flatten() - cy, xx.flatten() - cx)
    amp_flat = amp_mean.flatten()

    r_n = r / (r.max() + 1e-8)
    theta_n = (theta + np.pi) / (2 * np.pi)
    amp_n = np.log1p(amp_flat)
    amp_n = amp_n / (amp_n.max() + 1e-8)

    features = np.stack([r_n, theta_n, amp_n], axis=1)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(features)
    return km.labels_.reshape(H, W), features


def main(frames_root, out_dir):
    frames_root = Path(frames_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fake_types = ["Deepfakes", "Face2Face", "NeuralTextures"]
    amps = {}

    print("[1] 진폭 스펙트럼 계산")
    for ft in fake_types:
        amps[ft] = mean_amplitude_spectrum(frames_root / ft)

    # ─────────────────────────────────────────
    # (A) 진폭 스펙트럼 자체가 유형별로 얼마나 다른가
    # ─────────────────────────────────────────
    print("\n[2] 진폭 스펙트럼 차이 (정규화 후 MAE)")
    norm_amps = {}
    for ft in fake_types:
        a = amps[ft]
        a_log = np.log1p(a)
        norm_amps[ft] = (a_log - a_log.min()) / (a_log.max() - a_log.min() + 1e-8)

    pairs = [("Deepfakes", "Face2Face"), ("Deepfakes", "NeuralTextures"), ("Face2Face", "NeuralTextures")]
    for a, b in pairs:
        diff = np.abs(norm_amps[a] - norm_amps[b]).mean()
        print(f"    {a} vs {b}: MAE = {diff:.5f}")

    # 차이 히트맵 저장
    for a, b in pairs:
        diff = np.abs(norm_amps[a] - norm_amps[b])
        diff_vis = (diff / diff.max() * 255).astype(np.uint8)
        diff_color = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(out_dir / f"diff_{a}_vs_{b}.png"), diff_color)

    # ─────────────────────────────────────────
    # (B) amp_n 의 값 범위 vs r_n, theta_n 범위 비교
    # ─────────────────────────────────────────
    print("\n[3] Feature 값 범위 비교 (K-means 입력)")
    for ft in fake_types:
        _, features = cluster_spectrum(amps[ft], k=8)
        r_n, theta_n, amp_n = features[:, 0], features[:, 1], features[:, 2]
        print(f"    {ft}:")
        print(f"        r_n     range = [{r_n.min():.3f}, {r_n.max():.3f}], std={r_n.std():.4f}")
        print(f"        theta_n range = [{theta_n.min():.3f}, {theta_n.max():.3f}], std={theta_n.std():.4f}")
        print(f"        amp_n   range = [{amp_n.min():.3f}, {amp_n.max():.3f}], std={amp_n.std():.4f}")

    # ─────────────────────────────────────────
    # (C) 위조 유형별 K-means labels 가 얼마나 같은가 (label 일치율)
    # ─────────────────────────────────────────
    print("\n[4] K-means 클러스터 모양 일치율 (라벨 번호는 다를 수 있으니 IoU 최대 매칭 후 비교)")
    labels = {}
    for ft in fake_types:
        labels[ft], _ = cluster_spectrum(amps[ft], k=8)

    def best_match_agreement(l1, l2, k=8):
        """l1의 각 클러스터를 l2에서 가장 많이 겹치는 클러스터에 매칭, 전체 일치율 계산"""
        total = l1.size
        agree = 0
        for c in range(k):
            mask1 = (l1 == c)
            # l2에서 mask1과 가장 많이 겹치는 클러스터 찾기
            best_overlap = 0
            for c2 in range(k):
                overlap = np.logical_and(mask1, l2 == c2).sum()
                best_overlap = max(best_overlap, overlap)
            agree += best_overlap
        return agree / total

    for a, b in pairs:
        agreement = best_match_agreement(labels[a], labels[b])
        print(f"    {a} vs {b}: 클러스터 모양 일치율 = {agreement*100:.1f}%")

    # 라벨 시각화 저장
    for ft in fake_types:
        lab_vis = (labels[ft].astype(np.float32) / 7 * 255).astype(np.uint8)
        lab_color = cv2.applyColorMap(lab_vis, cv2.COLORMAP_RAINBOW)
        cv2.imwrite(str(out_dir / f"labels_{ft}.png"), lab_color)

    print(f"\n[✓] 시각화 저장 위치: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.frames_root, args.out_dir)