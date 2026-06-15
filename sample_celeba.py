"""
img_align_celeba.zip에서 5000장을 랜덤 샘플링해서 별도 폴더에 복사하는 스크립트.

사용법:
    python sample_celeba.py --zip ./img_align_celeba.zip --out ./celeba_5000
    python sample_celeba.py --zip "C:/Users/User/OneDrive/2024_부산대/바탕 화면/2024/공부 자료/26-1_생성모델_전상률/term_project/dataset/img_align_celeba.zip" --out ./celeba_5000 --seed 42
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path


def sample_celeba(zip_path: str, out_dir: str, n: int = 5000, seed: int = 42):
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] ZIP 파일 열기: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # .jpg 파일만 추출 (디렉토리 항목 제외)
        all_imgs = [
            name for name in zf.namelist()
            if name.lower().endswith(".jpg") and not name.endswith("/")
        ]
        print(f"      전체 이미지 수: {len(all_imgs):,}장")

        if len(all_imgs) < n:
            raise ValueError(f"ZIP 안에 이미지가 {len(all_imgs)}장뿐입니다 ({n}장 요청).")

        # 재현 가능한 랜덤 샘플링
        random.seed(seed)
        sampled = random.sample(all_imgs, n)
        sampled.sort()  # 파일명 순 정렬 (선택사항)

        print(f"[2/3] {n}장 샘플링 후 압축 해제 중...")
        for i, name in enumerate(sampled, 1):
            # ZIP 내부 경로에서 파일명만 추출
            fname = Path(name).name
            target = out_dir / fname

            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

            if i % 500 == 0 or i == n:
                print(f"      {i}/{n} 완료...")

    print(f"[3/3] 완료! 저장 위치: {out_dir.resolve()}")
    print(f"      파일 수 확인: {len(list(out_dir.glob('*.jpg')))}장")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip",  required=True,  help="img_align_celeba.zip 경로")
    parser.add_argument("--out",  required=True,  help="출력 폴더 경로")
    parser.add_argument("--n",    type=int, default=5000, help="샘플링 장수 (기본 5000)")
    parser.add_argument("--seed", type=int, default=42,   help="랜덤 시드 (기본 42)")
    args = parser.parse_args()

    sample_celeba(args.zip, args.out, args.n, args.seed)