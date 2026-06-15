"""
CelebA-HQ 5,000장에서 Type-A / Type-B 조건 맵을 생성하는 스크립트.

마스크 B = ring [0,1,2] (저주파, 256x256 기준)을 사용.
- Type-A (억제 맵): 이미지 A의 저주파 진폭을 0으로 제거
- Type-B (교체 맵): 이미지 A의 저주파 진폭을 랜덤 페어 이미지 B의 저주파 진폭으로 교체
- 목표 이미지(target): A 원본 그대로 (두 타입 공통, 512x512)

출력 구조:
    out_dir/
    ├── targets/         원본 이미지 (512x512, 모든 타입 공통)
    │   ├── 000001.png
    │   └── ...
    ├── typeA/            Type-A 조건 맵 (512x512)
    │   ├── 000001.png
    │   └── ...
    └── typeB/            Type-B 조건 맵 (512x512)
        ├── 000001.png
        └── ...

사용법:
    python make_condition_maps.py \
        --celeba_dir "C:/Temp/TP2606_deepfake/celeba_5000" \
        --mask_path "C:/Temp/TP2606_deepfake/masks/mask_B_Deepfakes.npy" \
        --out_dir "C:/Temp/TP2606_deepfake/conditions" \
        --image_size 512 \
        --fft_size 256
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_and_resize(path: Path, size: int):
    """이미지를 읽어서 RGB, size x size로 리사이즈 (0~255 uint8)."""
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size))


def apply_freq_transform(img_rgb: np.ndarray, B_mask: np.ndarray, mode: str,
                          fft_size: int, pair_rgb: np.ndarray = None):
    """
    img_rgb: (size, size, 3) uint8, 변환 대상 이미지 A
    B_mask: (fft_size, fft_size) 0/1, 저주파 마스크
    mode: "A" (억제) 또는 "B" (교체, pair_rgb 필요)
    pair_rgb: (size, size, 3) uint8, Type-B용 페어 이미지

    반환: (size, size, 3) uint8, 변환된 조건 맵
    """
    size = img_rgb.shape[0]
    out = np.zeros_like(img_rgb, dtype=np.float32)

    for ch in range(3):
        ch_img = img_rgb[:, :, ch].astype(np.float32)
        ch_resized = cv2.resize(ch_img, (fft_size, fft_size))

        fft = np.fft.fftshift(np.fft.fft2(ch_resized))
        amp = np.abs(fft)
        phase = np.angle(fft)

        if mode == "A":
            # 저주파 진폭 제거
            amp_new = amp * (1.0 - B_mask)
        elif mode == "B":
            # 저주파 진폭을 페어 이미지의 것으로 교체
            pair_ch = pair_rgb[:, :, ch].astype(np.float32)
            pair_resized = cv2.resize(pair_ch, (fft_size, fft_size))
            pair_fft = np.fft.fftshift(np.fft.fft2(pair_resized))
            pair_amp = np.abs(pair_fft)
            amp_new = amp * (1.0 - B_mask) + pair_amp * B_mask
        else:
            raise ValueError(f"unknown mode: {mode}")

        fft_new = amp_new * np.exp(1j * phase)
        img_back = np.fft.ifft2(np.fft.ifftshift(fft_new)).real

        # fft_size -> size 복원
        img_back = cv2.resize(img_back.astype(np.float32), (size, size))
        # IFFT 결과는 원래 거의 0~255 범위이므로 clip만 적용
        # (min-max 정규화를 적용하면 진폭 변화로 생긴 미세한 차이가
        #  재스트레칭 과정에서 거의 상쇄되어 변환 효과가 사라짐)
        out[:, :, ch] = np.clip(img_back, 0, 255)

    return out.astype(np.uint8)


def main(args):
    celeba_dir = Path(args.celeba_dir)
    out_dir = Path(args.out_dir)
    targets_dir = out_dir / "targets"
    typeA_dir = out_dir / "typeA"
    typeB_dir = out_dir / "typeB"
    for d in [targets_dir, typeA_dir, typeB_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 마스크 로드
    B_mask = np.load(args.mask_path)
    print(f"[*] 마스크 로드: {args.mask_path}, shape={B_mask.shape}, 선택비율={B_mask.mean()*100:.1f}%")

    # 이미지 목록
    image_paths = sorted(celeba_dir.glob("*.jpg")) + sorted(celeba_dir.glob("*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    n = len(image_paths)
    print(f"[*] 총 이미지: {n}장")

    # Type-B용 랜덤 페어 인덱스 (자기 자신과 페어되지 않도록)
    random.seed(args.seed)
    indices = list(range(n))
    pair_indices = indices[:]
    random.shuffle(pair_indices)
    for i in range(n):
        if pair_indices[i] == i:
            # 자기 자신과 겹치면 다음 인덱스와 교환
            swap_with = (i + 1) % n
            pair_indices[i], pair_indices[swap_with] = pair_indices[swap_with], pair_indices[i]

    print(f"[*] Type-B 랜덤 페어링 완료 (seed={args.seed})")

    for i, path in enumerate(tqdm(image_paths, desc="조건 맵 생성")):
        img = load_and_resize(path, args.image_size)
        fname = f"{i:06d}.png"

        # target (원본 그대로)
        cv2.imwrite(str(targets_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Type-A
        cond_a = apply_freq_transform(img, B_mask, mode="A", fft_size=args.fft_size)
        cv2.imwrite(str(typeA_dir / fname), cv2.cvtColor(cond_a, cv2.COLOR_RGB2BGR))

        # Type-B (페어 이미지 사용)
        pair_path = image_paths[pair_indices[i]]
        pair_img = load_and_resize(pair_path, args.image_size)
        cond_b = apply_freq_transform(img, B_mask, mode="B", fft_size=args.fft_size, pair_rgb=pair_img)
        cv2.imwrite(str(typeB_dir / fname), cv2.cvtColor(cond_b, cv2.COLOR_RGB2BGR))

    print(f"\n[✓] 완료!")
    print(f"    targets : {targets_dir} ({n}장)")
    print(f"    typeA   : {typeA_dir} ({n}장)")
    print(f"    typeB   : {typeB_dir} ({n}장)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeba_dir",  required=True, help="CelebA-HQ 5000장 폴더")
    parser.add_argument("--mask_path",   required=True, help="마스크 B .npy 경로")
    parser.add_argument("--out_dir",     required=True, help="출력 폴더")
    parser.add_argument("--image_size",  type=int, default=512, help="출력 이미지 크기 (기본 512)")
    parser.add_argument("--fft_size",    type=int, default=256, help="FFT 해상도 (기본 256, 마스크와 일치해야 함)")
    parser.add_argument("--max_images",  type=int, default=None, help="처리할 최대 이미지 수 (디버그용)")
    parser.add_argument("--seed",        type=int, default=42, help="Type-B 랜덤 페어링 시드")
    args = parser.parse_args()
    main(args)