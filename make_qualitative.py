"""
Exp-0/1/2에서 샘플을 골라 3행×N열 비교 그리드 이미지 생성.

사용법:
    # 기본 (seed로 랜덤 선택, 어두운 이미지 자동 제외)
    python make_qualitative.py \
        --generated_dir "C:/Temp/TP2606_deepfake/generated" \
        --out_path "C:/Temp/TP2606_deepfake/evaluation/qualitative.png" \
        --n_samples 3 \
        --seed 7

    # seed 바꿔서 다른 조합 시도
    python make_qualitative.py ... --seed 42

    # 인덱스 직접 지정 (각 행마다 원하는 파일 번호 지정, 쉼표로 구분)
    # 예: Exp-0은 00010,00025,00050 / Exp-1은 00003,00007,00012 / Exp-2는 00001,00020,00030
    python make_qualitative.py ... \
        --exp0_idx "00010,00025,00050" \
        --exp1_idx "00003,00007,00012" \
        --exp2_idx "00001,00020,00030"

    # 어두운 이미지 제외 임계값 조정 (기본 30, 낮출수록 덜 엄격)
    python make_qualitative.py ... --brightness_threshold 40

    # 프리뷰 모드: 각 폴더에서 후보 이미지 인덱스와 밝기를 출력
    python make_qualitative.py ... --preview
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def is_dark(img_bgr: np.ndarray, threshold: int = 30) -> bool:
    """평균 밝기가 threshold 이하면 어두운 이미지로 판단."""
    return img_bgr.mean() < threshold


def load_samples_auto(folder: Path, n: int, seed: int, brightness_threshold: int):
    """
    랜덤 샘플링 + 어두운 이미지 자동 제외.
    어두운 이미지가 많아 n장을 못 채우면 경고 출력.
    """
    paths = sorted(folder.glob("*.png"))
    random.seed(seed)
    random.shuffle(paths)

    selected = []
    skipped = []
    for p in paths:
        if len(selected) >= n:
            break
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        if is_dark(bgr, brightness_threshold):
            skipped.append(p.stem)
            continue
        selected.append((p.stem, cv2.resize(bgr, (256, 256))))

    if skipped:
        print(f"    [{folder.name}] 어두운 이미지 {len(skipped)}장 제외: {skipped[:5]}{'...' if len(skipped)>5 else ''}")
    if len(selected) < n:
        print(f"    [경고] {folder.name}: {n}장 요청했지만 {len(selected)}장만 확보됨.")

    return selected


def load_samples_manual(folder: Path, indices: list):
    """
    사용자가 지정한 인덱스(파일명 앞부분)로 직접 선택.
    예: indices = ["00010", "00025", "00050"]
    """
    selected = []
    for idx in indices:
        # 파일명이 정확히 매칭되는 것 찾기
        matches = list(folder.glob(f"{idx}*.png"))
        if not matches:
            print(f"    [경고] {folder.name}/{idx}*.png 파일을 찾을 수 없습니다.")
            continue
        p = matches[0]
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"    [경고] 이미지 로드 실패: {p}")
            continue
        selected.append((p.stem, cv2.resize(bgr, (256, 256))))
    return selected


def preview_mode(generated_dir: Path, brightness_threshold: int):
    """각 폴더의 이미지 밝기 정보를 출력해 인덱스 선택을 돕는 모드."""
    exps = [
        ("exp0_baseline", generated_dir / "exp0_baseline"),
        ("exp1_typeA",    generated_dir / "exp1_typeA"),
        ("exp2_typeB",    generated_dir / "exp2_typeB"),
    ]
    print("\n=== 프리뷰 모드: 각 폴더의 이미지 밝기 (상위 30장) ===")
    for name, folder in exps:
        paths = sorted(folder.glob("*.png"))[:30]
        print(f"\n[{name}]")
        print(f"  {'인덱스':<10} {'평균밝기':>8}  {'상태':>8}")
        print(f"  {'-'*30}")
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            brightness = bgr.mean()
            status = "❌ 어두움" if brightness < brightness_threshold else "✅ 사용가능"
            print(f"  {p.stem:<10} {brightness:>8.1f}  {status}")
    print("\n위 인덱스를 참고해서 --exp0_idx, --exp1_idx, --exp2_idx 옵션으로 지정하세요.")
    print("예시: --exp1_idx \"00003,00007,00012\"")


def main(args):
    generated_dir = Path(args.generated_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 프리뷰 모드
    if args.preview:
        preview_mode(generated_dir, args.brightness_threshold)
        return

    exps = [
        ("Exp-0\n(Baseline)", generated_dir / "exp0_baseline", args.exp0_idx),
        ("Exp-1\n(Type-A)",   generated_dir / "exp1_typeA",    args.exp1_idx),
        ("Exp-2\n(Type-B)",   generated_dir / "exp2_typeB",    args.exp2_idx),
    ]

    n = args.n_samples
    cell_h, cell_w = 256, 256
    label_w = 110
    pad = 6

    # 각 행의 이미지 로드
    rows_data = []
    for label, exp_dir, manual_idx in exps:
        print(f"[{label.replace(chr(10), ' ')}]")
        if manual_idx:
            indices = [x.strip() for x in manual_idx.split(",")]
            samples = load_samples_manual(exp_dir, indices)
        else:
            samples = load_samples_auto(exp_dir, n, args.seed, args.brightness_threshold)
        rows_data.append((label, samples))
        names = [s[0] for s in samples]
        print(f"    선택된 파일: {names}")

    # 실제 n_samples는 가장 적은 행 기준
    actual_n = min(len(s) for _, s in rows_data)
    if actual_n == 0:
        print("[오류] 선택된 이미지가 없습니다.")
        return

    total_w = label_w + (cell_w + pad) * actual_n
    total_h = (cell_h + pad) * len(rows_data)
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX

    for row, (label, samples) in enumerate(rows_data):
        y0 = row * (cell_h + pad)
        for li, line in enumerate(label.split("\n")):
            cv2.putText(canvas, line, (4, y0 + 30 + li * 28),
                        font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        for col, (_, img) in enumerate(samples[:actual_n]):
            x0 = label_w + col * (cell_w + pad)
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = img

    cv2.imwrite(str(out_path), canvas)
    print(f"\n[✓] 저장: {out_path}  ({canvas.shape[1]}×{canvas.shape[0]}px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir",       required=True,  help="generated/ 폴더")
    parser.add_argument("--out_path",            required=True,  help="저장 경로 (.png)")
    parser.add_argument("--n_samples",           type=int, default=3, help="열 수 (기본 3)")
    parser.add_argument("--seed",                type=int, default=7,  help="랜덤 시드 (기본 7)")
    parser.add_argument("--brightness_threshold",type=int, default=30, help="어두운 이미지 제외 임계값 (기본 30)")
    parser.add_argument("--exp0_idx", default=None, help="Exp-0 파일 인덱스 (쉼표 구분, 예: '00010,00025,00050')")
    parser.add_argument("--exp1_idx", default=None, help="Exp-1 파일 인덱스")
    parser.add_argument("--exp2_idx", default=None, help="Exp-2 파일 인덱스")
    parser.add_argument("--preview",  action="store_true", help="각 폴더 이미지 밝기 프리뷰 출력")
    args = parser.parse_args()
    main(args)