
# 실험 목표

탐지기가 의존하는 지배적 주파수 대역을 ControlNet 조건으로 활용해, 그 주파수 특성이 교란된(혹은 실제 이미지처럼 보정된) 얼굴 이미지를 Stable Diffusion으로 생성 — 탐지를 회피할 수 있는지 검증.


# Pipeline Steps

1. `extract_frames.py` — FF++ 비디오에서 프레임 추출
2. `extract_freq_mask.py` — 주파수 마스크 B 추출
3. `make_ring_mask.py` — Ring 마스크 직접 생성
4. `make_condition_maps.py` — Type-A/B 조건 맵 생성
5. `prepare_dataset.py` — ControlNet 학습용 데이터셋 준비
6. `train_controlnet.py` — ControlNet 학습 (diffusers 공식 스크립트)
7. `generate.py` — 이미지 생성 (Exp-0/1/2)
8. `evaluate.py` — 평가 (FID / 주파수 거리 / 회피율)
9. `make_qualitative.py` — 정성적 비교 이미지 생성
10. `make_results_chart.py` — 결과 차트 생성


## Pretrained Weights
ControlNet-A and ControlNet-B weights are not included in this repository due to file size (13GB each).
To reproduce the results, train from scratch using the provided scripts (see Step 5-6 above),
or contact the authors for access.


---------

# 실험 셋팅

```
# Python 환경
conda create -n freqcontrolnet python=3.10
conda activate freqcontrolnet

# 핵심 패키지
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
pip install opencv-python scikit-learn scipy
pip install pytorch-fid  # FID 평가용
pip install numpy timm

# Xception 가중치 다운로드
git clone https://github.com/SCLBD/DeepfakeBench.git
# [pre-trained weights]에서 xception_best.pth 다운받아서 DeepfakeBench\training\weights 아래에 저장


# diffusers ControlNet 학습 스크립트
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
cd examples/controlnet
pip install -r requirements.txt

```


# 데이터셋 준비

- CelebA-HQ: Align&Cropped = 실제 이미지 
    - python sample_celeba.py --zip "C:/Users/.../img_align_celeba.zip" --out ./celeba_5000 --seed 42
        - 5000개 샘플링

- FaceForensics++ = 딥페이크 이미지 (faceforensics_download_v4.py는 신청해서 받아야 함)
    - python faceforensics_download_v4.py "C:\Temp\TP2606_deepfake\FFHQ_pp" -d Deepfakes -c c23 -t videos -n 200 --server EU2
        - 얼굴 전체 교체
    - python faceforensics_download_v4.py "C:\Temp\TP2606_deepfake\FFHQ_pp" -d Face2Face -c c23 -t videos -n 200 --server EU2
        - 표정만 이식
    - python faceforensics_download_v4.py "C:\Temp\TP2606_deepfake\FFHQ_pp" -d NeuralTextures -c c23 -t videos -n 200 --server EU2
        - 텍스처 재렌더링

- FF++의 비디오 데이터를 프레임으로 나누기 (유형별로 4000장씩)
    - python extract_frames.py \
        --video_dir "C:/Temp/TP2606_deepfake/FFHQ_pp/manipulated_sequences/Deepfakes/c23/videos" \
        --out_dir "C:/Temp/TP2606_deepfake/frames/Deepfakes" \
        --frames_per_clip 20

- 테스트 데이터셋: 


# 마스크 추출

- python extract_freq_mask.py --frames_dir "C:/Temp/TP2606_deepfake/frames/Deepfakes" --weights_path "C:/Temp/TP2606_deepfake/DeepfakeBench/training/weights/xception_best.pth" --out_dir "C:/Temp/TP2606_deepfake/masks" --fake_type Deepfakes

- python extract_freq_mask.py --frames_dir "C:/Temp/TP2606_deepfake/frames/Face2Face" --weights_path "C:/Temp/TP2606_deepfake/DeepfakeBench/training/weights/xception_best.pth" --out_dir "C:/Temp/TP2606_deepfake/masks" --fake_type Face2Face

- python extract_freq_mask.py --frames_dir "C:/Temp/TP2606_deepfake/frames/NeuralTextures" --weights_path "C:/Temp/TP2606_deepfake/DeepfakeBench/training/weights/xception_best.pth" --out_dir "C:/Temp/TP2606_deepfake/masks" --fake_type NeuralTextures


### 마스크 생성 과정 요약
1. amp_mean = 2000장의 그레이스케일 FFT 진폭 스펙트럼 평균 (256×256)
2. K-means(k=8)로 (r, θ, amp) 기반 클러스터링 → labels (256×256)
   ← 이 labels는 각 위조 유형마다 "따로" 계산됨 (입력 이미지가 다르므로)
3. 클러스터별로 OHEM: 그 클러스터를 억제했을 때 탐지기 손실
4. 손실 낮은 top-3 클러스터 → 마스크 B

### 탐지기가 어떤 주파수 대역에 의존하는가
ring [0,1,2] (저주파 38.7%)를 세 유형 공통 마스크 B로 확정. 
세 유형이 동일한 마스크를 사용하더라도, 핵심 연구질문("ControlNet이 주파수 조건을 학습 가능한가")은 검증 가능




# 조건맵 생성

- python make_condition_maps2.py --celeba_dir "C:/Temp/TP2606_deepfake/celeba_5000" --mask_path "C:/Temp/TP2606_deepfake/masks/mask_B_Deepfakes.npy" --out_dir "C:/Temp/TP2606_deepfake/conditions" --image_size 512 --fft_size 256
    - Type-A: A의 ring[0,1,2](저주파) 진폭 → 0으로 제거
    - Type-B: A의 ring[0,1,2](저주파) 진폭 → B의 해당 진폭으로 교체



# HuggingFace 데이터셋 형식 맞추기
- python prepare_dataset.py --conditions_dir "C:/Temp/TP2606_deepfake/conditions" --out_dir "C:/Temp/TP2606_deepfake/dataset_typeA" --cond_type typeA --prompt "a photo of a human face"

- python prepare_dataset.py --conditions_dir "C:/Temp/TP2606_deepfake/conditions" --out_dir "C:/Temp/TP2606_deepfake/dataset_typeB" --cond_type typeB --prompt "a photo of a human face"




# 학습
- cd C:\Temp\TP2606_deepfake\diffusers\examples\controlnet

- accelerate launch train_controlnet.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --output_dir="C:/Temp/TP2606_deepfake/controlnet_typeA" --train_data_dir="C:/Temp/TP2606_deepfake/dataset_typeA" --image_column="image" --conditioning_image_column="conditioning_image" --caption_column="text" --resolution=512 --learning_rate=1e-5 --train_batch_size=2 --gradient_accumulation_steps=2 --num_train_epochs=20 --mixed_precision="fp16" --gradient_checkpointing --checkpointing_steps=2000 --validation_steps=1000

- 뭔가 에러가 나서 조치 취하고

- python train_controlnet.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --output_dir="C:/Temp/TP2606_deepfake/controlnet_typeA" --train_data_dir="C:/Temp/TP2606_deepfake/dataset_typeA" --image_column="image" --conditioning_image_column="conditioning_image" --caption_column="text" --resolution=512 --learning_rate=1e-5 --train_batch_size=2 --gradient_accumulation_steps=2 --num_train_epochs=20 --mixed_precision="fp16" --gradient_checkpointing --checkpointing_steps=2000 --validation_steps=1000

- 너무 오래 걸려서 학습 파라미터 바꿈
- typeA
    - python train_controlnet.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --output_dir="C:/Temp/TP2606_deepfake/controlnet_typeA" --train_data_dir="C:/Temp/TP2606_deepfake/dataset_typeA" --image_column="image" --conditioning_image_column="conditioning_image" --caption_column="text" --resolution=512 --learning_rate=1e-5 --train_batch_size=2 --gradient_accumulation_steps=2 --num_train_epochs=8 --max_train_steps=6000 --mixed_precision="fp16" --gradient_checkpointing --checkpointing_steps=2000 --validation_steps=2000
- typeB
    - python train_controlnet.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --output_dir="C:/Temp/TP2606_deepfake/controlnet_typeB" --train_data_dir="C:/Temp/TP2606_deepfake/dataset_typeB" --image_column="image" --conditioning_image_column="conditioning_image" --caption_column="text" --resolution=512 --learning_rate=1e-5 --train_batch_size=2 --gradient_accumulation_steps=2 --num_train_epochs=8 --max_train_steps=6000 --mixed_precision="fp16" --gradient_checkpointing --checkpointing_steps=2000 --validation_steps=2000




# 이미지 생성
- python generate.py --controlnet_A "C:/Temp/TP2606_deepfake/controlnet_typeA" --controlnet_B "C:/Temp/TP2606_deepfake/controlnet_typeB" --cond_A_dir "C:/Temp/TP2606_deepfake/conditions/typeA" --cond_B_dir "C:/Temp/TP2606_deepfake/conditions/typeB" --out_dir "C:/Temp/TP2606_deepfake/generated" --num_images 500




# 평가
pip install pytorch-fid

python evaluate.py --generated_dir "C:/Temp/TP2606_deepfake/generated" --celeba_dir "C:/Temp/TP2606_deepfake/celeba_5000" --weights_path "C:/Temp/TP2606_deepfake/DeepfakeBench/training/weights/xception_best.pth" --mask_path "C:/Temp/TP2606_deepfake/masks/mask_B_lowfreq012.npy" --out_dir "C:/Temp/TP2606_deepfake/evaluation"


python make_qualitative.py --generated_dir "C:/Temp/TP2606_deepfake/generated" --out_path "C:/Temp/TP2606_deepfake/evaluation/qualitative.png" --exp0_idx "00078,00085,00087" --exp1_idx "00078,00085,00087" --exp2_idx "00078,00085,00087"


pip install matplotlib
python make_results_chart.py --results_json "C:/Temp/TP2606_deepfake/evaluation/results.json" --out_dir "C:/Temp/TP2606_deepfake/evaluation/charts"