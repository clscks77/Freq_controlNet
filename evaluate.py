"""
세 가지 실험 조건(Exp-0/1/2)에 대해 평가를 수행하는 스크립트.

평가 지표:
  1. FID (Frechet Inception Distance): 생성 이미지 vs CelebA-HQ 실제 이미지
  2. 주파수 거리 (Freq. Dist.): Ring [0,1,2] 진폭 MAE, 생성 이미지 vs CelebA-HQ
  3. 회피율 (Evasion Rate): Xception 탐지기가 생성 이미지를 real로 분류하는 비율

사용법:
    python evaluate.py \
        --generated_dir "C:/Temp/TP2606_deepfake/generated" \
        --celeba_dir "C:/Temp/TP2606_deepfake/celeba_5000" \
        --weights_path "C:/Temp/TP2606_deepfake/DeepfakeBench/training/weights/xception_best.pth" \
        --mask_path "C:/Temp/TP2606_deepfake/masks/mask_B_lowfreq012.npy" \
        --out_dir "C:/Temp/TP2606_deepfake/evaluation"
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


# ──────────────────────────────────────────────
# 1. Xception 로드 (마스크 추출 때와 동일)
# ──────────────────────────────────────────────

def load_xception(weights_path: str, device: torch.device):
    try:
        import timm
    except ImportError:
        os.system(f"{sys.executable} -m pip install timm -q")
        import timm

    model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        elif k.startswith("head.fc."):
            k = k[len("head."):]
        k = k.replace("last_linear.", "fc.")
        new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model.eval().to(device)
    print(f"[✓] Xception 로드 완료")
    return model


# ──────────────────────────────────────────────
# 2. 이미지 로드 유틸
# ──────────────────────────────────────────────

def load_images_rgb(folder: Path, size: int = 299, max_n: int = None):
    """폴더에서 이미지를 읽어 RGB numpy list 반환."""
    paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if max_n:
        paths = paths[:max_n]
    imgs = []
    for p in tqdm(paths, desc=f"  로드 {folder.name}", leave=False):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size))
        imgs.append(rgb)
    return imgs


# ──────────────────────────────────────────────
# 3. FID 계산
# ──────────────────────────────────────────────

def compute_fid(gen_dir: Path, ref_dir: Path, device: torch.device, n: int = 500):
    """pytorch-fid 라이브러리 사용."""
    try:
        from pytorch_fid import fid_score
    except ImportError:
        os.system(f"{sys.executable} -m pip install pytorch-fid -q")
        from pytorch_fid import fid_score

    # pytorch-fid는 폴더 경로를 직접 받음
    fid = fid_score.calculate_fid_given_paths(
        [str(gen_dir), str(ref_dir)],
        batch_size=32,
        device=device,
        dims=2048,
        num_workers=0,
    )
    return fid


# ──────────────────────────────────────────────
# 4. 주파수 거리 (Ring [0,1,2] 진폭 MAE)
# ──────────────────────────────────────────────

def mean_ring_amplitude(imgs_rgb: list, B_mask: np.ndarray, fft_size: int = 256):
    """
    이미지 리스트에 대해 마스크 B 영역의 평균 FFT 진폭을 반환.
    """
    amp_sum = 0.0
    n = 0
    for rgb in imgs_rgb:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_r = cv2.resize(gray, (fft_size, fft_size))
        fft = np.fft.fftshift(np.fft.fft2(gray_r))
        amp = np.abs(fft)
        # 마스크 영역 평균 진폭
        amp_sum += (amp * B_mask).sum() / (B_mask.sum() + 1e-8)
        n += 1
    return amp_sum / max(n, 1)


def compute_freq_distance(gen_imgs: list, ref_imgs: list, B_mask: np.ndarray):
    """생성 이미지와 실제 이미지의 Ring 진폭 MAE."""
    gen_amp = mean_ring_amplitude(gen_imgs, B_mask)
    ref_amp = mean_ring_amplitude(ref_imgs, B_mask)
    return abs(gen_amp - ref_amp), gen_amp, ref_amp


# ──────────────────────────────────────────────
# 5. 회피율 (Evasion Rate)
# ──────────────────────────────────────────────

preprocess_xception = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def compute_evasion_rate(imgs_rgb: list, model, device: torch.device, batch_size: int = 32):
    """
    Xception이 생성 이미지를 real(class=0)로 분류하는 비율.
    DeepfakeBench 컨벤션: 0=real, 1=fake
    """
    real_count = 0
    total = 0

    for start in range(0, len(imgs_rgb), batch_size):
        batch_rgb = imgs_rgb[start: start + batch_size]
        tensors = torch.stack([preprocess_xception(img) for img in batch_rgb]).to(device)

        with torch.no_grad():
            logits = model(tensors)
            preds = logits.argmax(dim=1)  # 0=real, 1=fake
            real_count += (preds == 0).sum().item()
            total += len(batch_rgb)

    return real_count / max(total, 1)


# ──────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 디바이스: {device}\n")

    generated_dir = Path(args.generated_dir)
    celeba_dir = Path(args.celeba_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 마스크 로드
    B_mask = np.load(args.mask_path)
    print(f"[*] 마스크: {args.mask_path}, coverage={B_mask.mean()*100:.1f}%\n")

    # Xception 로드
    model = load_xception(args.weights_path, device)

    # 실험 폴더 목록
    experiments = {
        "Exp-0 (Baseline)": generated_dir / "exp0_baseline",
        "Exp-1 (Type-A)":   generated_dir / "exp1_typeA",
        "Exp-2 (Type-B)":   generated_dir / "exp2_typeB",
    }

    # CelebA-HQ 레퍼런스 이미지 로드 (FID용 폴더는 그대로, 주파수/회피율용은 500장)
    print("[*] CelebA-HQ 레퍼런스 이미지 로드 중 (500장)...")
    ref_imgs = load_images_rgb(celeba_dir, size=299, max_n=500)
    print(f"    {len(ref_imgs)}장 로드 완료\n")

    results = {}

    for exp_name, exp_dir in experiments.items():
        print(f"{'='*50}")
        print(f"[{exp_name}]  {exp_dir}")
        print(f"{'='*50}")

        # 이미지 로드
        gen_imgs = load_images_rgb(exp_dir, size=299)
        print(f"  생성 이미지: {len(gen_imgs)}장")

        # (1) FID
        print("  [1/3] FID 계산 중...")
        fid = compute_fid(exp_dir, celeba_dir, device)
        print(f"        FID = {fid:.4f}")

        # (2) 주파수 거리
        print("  [2/3] 주파수 거리 계산 중...")
        freq_dist, gen_amp, ref_amp = compute_freq_distance(gen_imgs, ref_imgs, B_mask)
        print(f"        Ring[0,1,2] 진폭 — 생성: {gen_amp:.4f}, 실제: {ref_amp:.4f}, MAE: {freq_dist:.4f}")

        # (3) 회피율
        print("  [3/3] 회피율 계산 중...")
        evasion = compute_evasion_rate(gen_imgs, model, device)
        print(f"        Evasion Rate = {evasion*100:.2f}%")

        results[exp_name] = {
            "FID": fid,
            "Freq_Dist_MAE": freq_dist,
            "Gen_Amp": gen_amp,
            "Ref_Amp": ref_amp,
            "Evasion_Rate": evasion,
        }
        print()

    # 결과 요약 출력
    print("\n" + "="*60)
    print("최종 결과 요약")
    print("="*60)
    print(f"{'실험':<22} {'FID':>8} {'Freq Dist':>12} {'Evasion Rate':>14}")
    print("-"*60)
    for exp_name, r in results.items():
        print(f"{exp_name:<22} {r['FID']:>8.4f} {r['Freq_Dist_MAE']:>12.4f} {r['Evasion_Rate']*100:>13.2f}%")

    # 결과 저장
    import json
    result_path = out_dir / "results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] 결과 저장: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", required=True, help="generated/ 폴더 (exp0~2 하위폴더 포함)")
    parser.add_argument("--celeba_dir",    required=True, help="CelebA-HQ 이미지 폴더 (FID 레퍼런스)")
    parser.add_argument("--weights_path",  required=True, help="xception_best.pth 경로")
    parser.add_argument("--mask_path",     required=True, help="mask_B_lowfreq012.npy 경로")
    parser.add_argument("--out_dir",       required=True, help="결과 저장 폴더")
    args = parser.parse_args()
    main(args)