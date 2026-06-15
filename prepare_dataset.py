"""
conditions/{targets,typeA,typeB} 폴더를 diffusers ControlNet 학습 스크립트가
요구하는 형식(metadata.jsonl + image/conditioning_image)으로 변환.

각 조건 타입(A/B)마다 별도 데이터셋 폴더를 만듦:
    dataset_typeA/
    ├── metadata.jsonl
    ├── images/        ← targets 복사 (학습 목표)
    └── conditioning_images/  ← typeA 복사 (조건)

사용법:
    python prepare_dataset.py \
        --conditions_dir "C:/Temp/TP2606_deepfake/conditions" \
        --out_dir "C:/Temp/TP2606_deepfake/dataset_typeA" \
        --cond_type typeA \
        --prompt "a photo of a human face"

    python prepare_dataset.py \
        --conditions_dir "C:/Temp/TP2606_deepfake/conditions" \
        --out_dir "C:/Temp/TP2606_deepfake/dataset_typeB" \
        --cond_type typeB \
        --prompt "a photo of a human face"
"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def main(conditions_dir, out_dir, cond_type, prompt):
    conditions_dir = Path(conditions_dir)
    out_dir = Path(out_dir)

    # datasets의 imagefolder 로더는 metadata.jsonl이
    # 이미지(file_name)와 "같은 폴더" 안에 있어야 인식함.
    # 구조:
    #   out_dir/
    #     metadata.jsonl
    #     000000.png ...           (targets, file_name)
    #     conditioning_images/
    #       000000.png ...         (조건 이미지)
    #
    # --train_data_dir 은 out_dir 자체를 가리켜야 함.
    cond_out = out_dir / "conditioning_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_out.mkdir(parents=True, exist_ok=True)

    targets_dir = conditions_dir / "targets"
    cond_dir = conditions_dir / cond_type

    target_files = sorted(targets_dir.glob("*.png"))
    print(f"[*] 총 {len(target_files)}장 변환 중 (cond_type={cond_type})")

    metadata_path = out_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for tpath in tqdm(target_files, desc="변환"):
            fname = tpath.name
            cpath = cond_dir / fname
            if not cpath.exists():
                print(f"[경고] 조건 이미지 없음, 스킵: {cpath}")
                continue

            # 복사: target은 out_dir 바로 아래, 조건은 conditioning_images/ 아래
            shutil.copy(tpath, out_dir / fname)
            shutil.copy(cpath, cond_out / fname)

            entry = {
                "file_name": fname,                           # out_dir 기준 상대경로
                "conditioning_image_file_name": f"conditioning_images/{fname}",
                "text": prompt,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[✓] 완료: {out_dir}")
    print(f"    metadata.jsonl: {metadata_path}")
    print(f"    --train_data_dir=\"{out_dir}\"  (이 폴더 자체를 지정)")
    print(f"    --image_column=image  --conditioning_image_column=conditioning_image_file_name")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cond_type", required=True, choices=["typeA", "typeB"])
    parser.add_argument("--prompt", default="a photo of a human face")
    args = parser.parse_args()
    main(args.conditions_dir, args.out_dir, args.cond_type, args.prompt)