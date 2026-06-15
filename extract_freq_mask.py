"""
FF++ 프레임에서 주파수 마스크 B를 추출하는 스크립트.
위조 유형별로 탐지기(Xception)가 가장 의존하는 주파수 대역을 찾아냄.

사용법 (위조 유형별로 각각 실행):
    python extract_freq_mask.py \
        --frames_dir "C:/Temp/TP2606_deepfake/frames/Deepfakes" \
        --weights_path "C:/Temp/TP2606_deepfake/DeepfakeBench/training/weights/xception_best.pth" \
        --out_dir "C:/Temp/TP2606_deepfake/masks" \
        --fake_type Deepfakes

    python extract_freq_mask.py \
        --frames_dir "C:/Temp/TP2606_deepfake/frames/Face2Face" \
        --weights_path "C:/Temp/TP2606_deepfake/DeepfakeBench/training/weights/xception_best.pth" \
        --out_dir "C:/Temp/TP2606_deepfake/masks" \
        --fake_type Face2Face

    python extract_freq_mask.py \
        --frames_dir "C:/Temp/TP2606_deepfake/frames/NeuralTextures" \
        --weights_path "C:/Temp/TP2606_deepfake/DeepfakeBench/training/weights/xception_best.pth" \
        --out_dir "C:/Temp/TP2606_deepfake/masks" \
        --fake_type NeuralTextures
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


# ──────────────────────────────────────────────
# 1. Xception 모델 로드 (DeepfakeBench 방식)
# ──────────────────────────────────────────────

def load_xception(weights_path: str, device: torch.device):
    """
    DeepfakeBench의 xception_best.pth 로드.
    키 구조: backbone.conv1.weight ... + head.fc.weight 형식.
    timm Xception의 키 구조와 다르므로 backbone. 접두사를 제거하고 매핑.
    """
    try:
        import timm
    except ImportError:
        print("[!] timm 패키지가 없습니다. 설치합니다...")
        os.system(f"{sys.executable} -m pip install timm -q")
        import timm

    # timm Xception (legacy_xception): 키가 conv1.weight, block1.rep.0.weight ...
    model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)

    state = torch.load(weights_path, map_location=device, weights_only=False)

    # DeepfakeBench state_dict 키 정리
    # backbone.XXX → XXX  /  head.fc.XXX → fc.XXX  /  module. 제거
    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        elif k.startswith("head.fc."):
            k = k[len("head."):]   # fc.weight, fc.bias
        # DeepfakeBench는 last_linear, timm legacy_xception은 fc
        k = k.replace("last_linear.", "fc.")
        # adjust_channel 등 DeepfakeBench 전용 레이어는 무시됨 (strict=False)
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[✓] Xception 가중치 로드 완료: {weights_path}")
    if missing:
        print(f"    missing  keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"    unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.eval().to(device)
    return model


# ──────────────────────────────────────────────
# 2. 이미지 로드 및 전처리
# ──────────────────────────────────────────────

IMG_SIZE = 299  # Xception 입력 크기

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_frames(frames_dir: Path, max_frames: int = 2000):
    """
    클립별 하위폴더에서 PNG 프레임을 읽어 텐서 리스트로 반환.
    최대 max_frames장만 사용 (메모리 절약).
    """
    paths = []
    for clip_dir in sorted(frames_dir.iterdir()):
        if clip_dir.is_dir():
            for png in sorted(clip_dir.glob("*.png")):
                paths.append(png)

    # 균등 샘플링
    if len(paths) > max_frames:
        step = len(paths) / max_frames
        paths = [paths[int(i * step)] for i in range(max_frames)]

    print(f"[*] 프레임 로드: {len(paths)}장 (원본 총 {len(paths)}장)")

    imgs_raw = []   # (H, W, 3) uint8 — FFT용
    imgs_tensor = []  # 전처리된 텐서 — 탐지기 입력용

    for p in tqdm(paths, desc="이미지 로드"):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs_raw.append(rgb)
        imgs_tensor.append(preprocess(rgb))

    return imgs_raw, torch.stack(imgs_tensor)  # (N, 3, 299, 299)


# ──────────────────────────────────────────────
# 3. FFT 진폭 스펙트럼 평균 계산
# ──────────────────────────────────────────────

def mean_amplitude_spectrum(imgs_raw: list, fft_size: int = 256):
    """
    그레이스케일 FFT 진폭 스펙트럼의 평균을 반환.
    fft_size×fft_size 로 리사이즈 후 계산.
    반환: amp_mean (fft_size, fft_size) numpy array
    """
    amp_sum = np.zeros((fft_size, fft_size), dtype=np.float64)
    for rgb in imgs_raw:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (fft_size, fft_size)).astype(np.float32)
        fft = np.fft.fftshift(np.fft.fft2(gray_resized))
        amp = np.abs(fft)
        amp_sum += amp
    amp_mean = amp_sum / len(imgs_raw)
    return amp_mean  # (fft_size, fft_size)


# ──────────────────────────────────────────────
# 4. K-means 클러스터링
# ──────────────────────────────────────────────

def cluster_spectrum(amp_mean: np.ndarray, k: int = 8):
    """
    K-means 대신 반경(r) 기준 동심원 ring으로 주파수 영역을 분할.
    ring 0 = 중심(저주파) ... ring k-1 = 외곽(고주파).

    K-means + (r,theta,amp) 방식은 amp의 영향이 r/theta에 비해 작아
    클러스터 모양이 위조 유형과 무관하게 거의 동일해지는 문제가 있었음.
    Ring 분할은 위치만으로 결정되므로 이 문제가 원천적으로 발생하지 않고,
    유형별 차이는 다음 단계인 OHEM(탐지기 손실)에서 순수하게 반영됨.

    반환: labels (H, W) int array, 값은 0 ~ k-1 (ring index, 0=중심)
    """
    H, W = amp_mean.shape
    cy, cx = H // 2, W // 2

    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max()

    # k개의 동심원 ring으로 균등 분할 (반경 기준 균등 — 면적 기준 아님)
    edges = np.linspace(0, r_max, k + 1)
    labels = np.zeros((H, W), dtype=np.int64)
    for i in range(k):
        if i == k - 1:
            mask = (r >= edges[i]) & (r <= edges[i + 1])
        else:
            mask = (r >= edges[i]) & (r < edges[i + 1])
        labels[mask] = i

    print(f"[*] 반경 기준 ring 분할 완료 (k={k}, ring 0=저주파 중심 ~ ring {k-1}=고주파 외곽)")
    return labels


# ──────────────────────────────────────────────
# 5. OHEM: 탐지기 손실 기반 상위 클러스터 선택
# ──────────────────────────────────────────────

def compute_loss_for_mask(
    mask_np, imgs_tensor, model, device, batch_size, fft_size
):
    """주어진 (fft_size,fft_size) suppression mask(1=억제)를 모든 이미지에 적용 후 평균 CE loss 반환.
    mask_np가 전부 0이면 baseline(무억제) loss가 됨."""
    fake_label_full = torch.ones(batch_size, dtype=torch.long).to(device)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(imgs_tensor), batch_size):
        batch = imgs_tensor[start: start + batch_size].to(device)
        B = batch.shape[0]

        suppressed = []
        for img in batch:
            channels = []
            for ch in range(3):
                ch_img = img[ch]
                ch_np = ch_img.cpu().numpy()
                ch_resized = cv2.resize(ch_np, (fft_size, fft_size))
                fft = np.fft.fftshift(np.fft.fft2(ch_resized))
                amp = np.abs(fft)
                phase = np.angle(fft)
                amp_suppressed = amp * (1.0 - mask_np)
                fft_new = amp_suppressed * np.exp(1j * phase)
                img_back = np.fft.ifft2(np.fft.ifftshift(fft_new)).real
                img_back = cv2.resize(img_back.astype(np.float32), (299, 299))
                channels.append(torch.tensor(img_back))
            suppressed.append(torch.stack(channels))

        suppressed_t = torch.stack(suppressed).to(device)

        with torch.no_grad():
            logits = model(suppressed_t)
            lbl = fake_label_full[:B]
            loss = F.cross_entropy(logits, lbl).item()

        total_loss += loss
        n_batches += 1

    return total_loss / max(n_batches, 1)


def ohem_select_clusters(
    labels: np.ndarray,
    imgs_tensor: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    k: int = 8,
    top_t: int = 3,
    batch_size: int = 16,
    fft_size: int = 256,
):
    """
    각 ring을 억제했을 때 baseline(무억제) 대비 손실 증가량(delta)이 가장 큰
    top_t ring을 선택 → 탐지기가 상대적으로 가장 의존하는 주파수 대역.

    절대 손실값 기준 선택은 "원래 정보량이 많은 ring"이 항상 선택되는
    편향이 있어 위조 유형 간 차이가 거의 드러나지 않았음 (모두 ring 0,1,2).
    baseline 대비 delta를 보면 이 정보량 편향이 상쇄되어
    유형별 상대적 중요도 차이가 드러남.
    """
    H, W = labels.shape

    # baseline: 아무것도 억제하지 않은 손실
    zero_mask = np.zeros((H, W), dtype=np.float32)
    baseline_loss = compute_loss_for_mask(zero_mask, imgs_tensor, model, device, batch_size, fft_size)
    print(f"    [baseline] 억제 없음 손실: {baseline_loss:.4f}")

    cluster_losses = []
    for c in range(k):
        mask_np = (labels == c).astype(np.float32)
        avg_loss = compute_loss_for_mask(mask_np, imgs_tensor, model, device, batch_size, fft_size)
        delta = avg_loss - baseline_loss
        cluster_losses.append((c, avg_loss, delta))
        print(f"    Ring {c} (저주파→고주파 순서 {c}/{k-1}) 억제 시 손실: {avg_loss:.4f}  (Δ={delta:+.4f})")

    # delta(손실 증가량)가 클수록 그 ring이 탐지에 더 중요 (제거 시 더 큰 타격)
    cluster_losses.sort(key=lambda x: x[2], reverse=True)
    top_clusters = [c for c, _, _ in cluster_losses[:top_t]]
    print(f"[✓] 선택된 상위 ring (top {top_t}, baseline 대비 Δloss 기준, 0=저주파~{k-1}=고주파): {sorted(top_clusters)}")
    return top_clusters


# ──────────────────────────────────────────────
# 6. 이진 마스크 B 생성 및 저장
# ──────────────────────────────────────────────

def build_binary_mask(labels: np.ndarray, top_clusters: list):
    """선택된 클러스터들을 1로 하는 이진 마스크 반환."""
    B = np.zeros_like(labels, dtype=np.float32)
    for c in top_clusters:
        B[labels == c] = 1.0
    return B


def visualize_mask(amp_mean: np.ndarray, B: np.ndarray, out_path: Path):
    """진폭 스펙트럼과 마스크를 나란히 시각화하여 저장."""
    amp_log = np.log1p(amp_mean)
    amp_vis = (amp_log / amp_log.max() * 255).astype(np.uint8)
    amp_color = cv2.applyColorMap(amp_vis, cv2.COLORMAP_JET)

    mask_vis = (B * 255).astype(np.uint8)
    mask_color = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    # 마스크 오버레이
    overlay = amp_color.copy()
    overlay[B == 1] = [0, 255, 0]  # 선택된 대역: 초록색
    combined = np.hstack([amp_color, mask_color, overlay])
    cv2.imwrite(str(out_path), combined)
    print(f"[✓] 시각화 저장: {out_path}")


# ──────────────────────────────────────────────
# 7. 메인
# ──────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 디바이스: {device}")

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) 모델 로드
    model = load_xception(args.weights_path, device)

    # (2) 프레임 로드
    imgs_raw, imgs_tensor = load_frames(frames_dir, max_frames=args.max_frames)
    print(f"[*] 총 {len(imgs_raw)}장 로드 완료")

    # (3) 진폭 스펙트럼 평균
    print("[*] FFT 진폭 스펙트럼 계산 중...")
    amp_mean = mean_amplitude_spectrum(imgs_raw, fft_size=args.fft_size)

    # (4) 반경 기준 ring 분할 (K-means 대체)
    labels = cluster_spectrum(amp_mean, k=args.k)

    # (5) OHEM ring 선택
    print("[*] OHEM 클러스터 선택 중 (시간이 걸릴 수 있습니다)...")
    top_clusters = ohem_select_clusters(
        labels, imgs_tensor, model, device,
        k=args.k, top_t=args.top_t,
        batch_size=args.batch_size,
        fft_size=args.fft_size,
    )

    # (6) 이진 마스크 저장
    B = build_binary_mask(labels, top_clusters)
    mask_save_path = out_dir / f"mask_B_{args.fake_type}.npy"
    np.save(str(mask_save_path), B)
    print(f"[✓] 마스크 저장: {mask_save_path}")

    # (7) 시각화
    vis_path = out_dir / f"mask_B_{args.fake_type}_vis.png"
    visualize_mask(amp_mean, B, vis_path)

    print("\n=== 완료 ===")
    print(f"  마스크 (.npy): {mask_save_path}")
    print(f"  시각화 (.png): {vis_path}")
    print(f"  마스크 shape : {B.shape}")
    print(f"  선택 비율    : {B.mean()*100:.1f}% of spectrum")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir",   required=True,  help="프레임 폴더 (클립별 하위폴더 구조)")
    parser.add_argument("--weights_path", required=True,  help="xception_best.pth 경로")
    parser.add_argument("--out_dir",      required=True,  help="마스크 저장 폴더")
    parser.add_argument("--fake_type",    required=True,  help="위조 유형 이름 (Deepfakes / Face2Face / NeuralTextures)")
    parser.add_argument("--max_frames",   type=int, default=2000, help="사용할 최대 프레임 수 (기본 2000)")
    parser.add_argument("--fft_size",     type=int, default=256,  help="FFT 해상도 (기본 256)")
    parser.add_argument("--k",            type=int, default=8,    help="K-means 클러스터 수 (기본 8)")
    parser.add_argument("--top_t",        type=int, default=3,    help="선택할 상위 클러스터 수 (기본 3)")
    parser.add_argument("--batch_size",   type=int, default=16,   help="탐지기 배치 크기 (기본 16)")
    args = parser.parse_args()
    main(args)