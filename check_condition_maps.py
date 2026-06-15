"""
조건 맵 생성 결과 검증용 진단 스크립트.
target / typeA / typeB 이미지를 나란히 놓고, target과의 차이(diff)를 시각화.

사용법:
    python check_condition_maps.py \
        --conditions_dir "C:/Temp/TP2606_deepfake/conditions_test" \
        --index 000000 \
        --out_dir "C:/Temp/TP2606_deepfake/conditions_test/diag"
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main(conditions_dir, index, out_dir):
    conditions_dir = Path(conditions_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{index}.png"
    target = cv2.imread(str(conditions_dir / "targets" / fname))
    typeA = cv2.imread(str(conditions_dir / "typeA" / fname))
    typeB = cv2.imread(str(conditions_dir / "typeB" / fname))

    if target is None or typeA is None or typeB is None:
        print("[오류] 파일을 읽을 수 없습니다. 경로를 확인하세요.")
        print(f"  target: {conditions_dir / 'targets' / fname}")
        print(f"  typeA : {conditions_dir / 'typeA' / fname}")
        print(f"  typeB : {conditions_dir / 'typeB' / fname}")
        return

    # 픽셀 차이 통계
    diff_A = np.abs(target.astype(np.int16) - typeA.astype(np.int16))
    diff_B = np.abs(target.astype(np.int16) - typeB.astype(np.int16))

    print(f"[target vs typeA] mean diff = {diff_A.mean():.3f}, max diff = {diff_A.max()}")
    print(f"[target vs typeB] mean diff = {diff_B.mean():.3f}, max diff = {diff_B.max()}")

    diff_AB = np.abs(typeA.astype(np.int16) - typeB.astype(np.int16))
    print(f"[typeA vs typeB]  mean diff = {diff_AB.mean():.3f}, max diff = {diff_AB.max()}")

    # 차이를 시각화 (대비 증폭)
    def amplify(d, scale=5):
        d_vis = np.clip(d * scale, 0, 255).astype(np.uint8)
        return d_vis

    cv2.imwrite(str(out_dir / f"{index}_diff_A_amplified.png"), amplify(diff_A))
    cv2.imwrite(str(out_dir / f"{index}_diff_B_amplified.png"), amplify(diff_B))

    # 나란히 비교 이미지
    combined = np.hstack([target, typeA, typeB])
    cv2.imwrite(str(out_dir / f"{index}_target_A_B_sidebyside.png"), combined)

    print(f"\n[✓] 결과 저장: {out_dir}")
    print(f"    {index}_target_A_B_sidebyside.png  (원본 | TypeA | TypeB 나란히)")
    print(f"    {index}_diff_A_amplified.png       (target vs TypeA 차이, 5배 증폭)")
    print(f"    {index}_diff_B_amplified.png       (target vs TypeB 차이, 5배 증폭)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions_dir", required=True)
    parser.add_argument("--index", default="000000")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.conditions_dir, args.index, args.out_dir)