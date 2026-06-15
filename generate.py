"""
세 가지 실험 조건으로 얼굴 이미지 생성.

Exp-0: 조건 없는 베이스라인 (일반 SD v1.5)
Exp-1: ControlNet-A (Type-A 조건 맵 기반)
Exp-2: ControlNet-B (Type-B 조건 맵 기반)

각 500장 생성, DDIM 50 스텝, guidance_scale=7.5

사용법:
    python generate.py \
        --controlnet_A "C:/Temp/TP2606_deepfake/controlnet_typeA" \
        --controlnet_B "C:/Temp/TP2606_deepfake/controlnet_typeB" \
        --cond_A_dir "C:/Temp/TP2606_deepfake/conditions/typeA" \
        --cond_B_dir "C:/Temp/TP2606_deepfake/conditions/typeB" \
        --out_dir "C:/Temp/TP2606_deepfake/generated" \
        --num_images 500
"""

import argparse
import random
from pathlib import Path

import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
)
from PIL import Image
from tqdm import tqdm

PROMPT = "a photo of a human face"
NEG_PROMPT = "blurry, low quality, cartoon, drawing, artifact"
SD_MODEL = "runwayml/stable-diffusion-v1-5"


def make_sd_pipeline(device):
    """Exp-0: 조건 없는 베이스라인 파이프라인"""
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def make_controlnet_pipeline(controlnet_path, device):
    """Exp-1/2: ControlNet 파이프라인"""
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16
    ).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL, controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_baseline(pipe, out_dir, num_images, seed=42):
    """Exp-0: 조건 없이 생성"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    for i in tqdm(range(num_images), desc="Exp-0 (baseline)"):
        image = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512,
        ).images[0]
        image.save(out_dir / f"{i:05d}.png")

    print(f"[✓] Exp-0 완료: {out_dir} ({num_images}장)")


def generate_controlnet(pipe, cond_dir, out_dir, num_images, seed=42):
    """Exp-1/2: ControlNet 조건 이미지와 함께 생성"""
    cond_dir = Path(cond_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_paths = sorted(cond_dir.glob("*.png"))
    if len(cond_paths) < num_images:
        print(f"[경고] 조건 이미지 {len(cond_paths)}장 < 요청 {num_images}장, 전체 사용")
        num_images = len(cond_paths)

    # 랜덤 샘플링 (재현 가능)
    random.seed(seed)
    selected = random.sample(cond_paths, num_images)
    selected.sort()

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    for i, cond_path in enumerate(tqdm(selected, desc=f"생성 중 ({out_dir.name})")):
        cond_image = Image.open(cond_path).convert("RGB").resize((512, 512))
        image = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=cond_image,
            num_inference_steps=50,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            generator=generator,
            height=512,
            width=512,
        ).images[0]
        image.save(out_dir / f"{i:05d}.png")

    print(f"[✓] 완료: {out_dir} ({num_images}장)")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 디바이스: {device}")

    out_dir = Path(args.out_dir)

    # ── Exp-0: 베이스라인 ──────────────────────────
    print("\n[Exp-0] 베이스라인 생성 중...")
    pipe0 = make_sd_pipeline(device)
    generate_baseline(pipe0, out_dir / "exp0_baseline", args.num_images)
    del pipe0
    torch.cuda.empty_cache()

    # ── Exp-1: ControlNet-A ────────────────────────
    print("\n[Exp-1] ControlNet-A 생성 중...")
    pipe1 = make_controlnet_pipeline(args.controlnet_A, device)
    generate_controlnet(pipe1, args.cond_A_dir, out_dir / "exp1_typeA", args.num_images)
    del pipe1
    torch.cuda.empty_cache()

    # ── Exp-2: ControlNet-B ────────────────────────
    print("\n[Exp-2] ControlNet-B 생성 중...")
    pipe2 = make_controlnet_pipeline(args.controlnet_B, device)
    generate_controlnet(pipe2, args.cond_B_dir, out_dir / "exp2_typeB", args.num_images)
    del pipe2
    torch.cuda.empty_cache()

    print(f"\n=== 전체 생성 완료 ===")
    for exp in ["exp0_baseline", "exp1_typeA", "exp2_typeB"]:
        n = len(list((out_dir / exp).glob("*.png")))
        print(f"  {exp}: {n}장")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet_A", required=True, help="ControlNet-A 가중치 폴더")
    parser.add_argument("--controlnet_B", required=True, help="ControlNet-B 가중치 폴더")
    parser.add_argument("--cond_A_dir",   required=True, help="Type-A 조건 맵 폴더")
    parser.add_argument("--cond_B_dir",   required=True, help="Type-B 조건 맵 폴더")
    parser.add_argument("--out_dir",      required=True, help="생성 이미지 저장 폴더")
    parser.add_argument("--num_images",   type=int, default=500, help="생성할 이미지 수 (기본 500)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
    main(args)