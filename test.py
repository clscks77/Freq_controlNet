import torch

state = torch.load(
    "C:/Temp/TP2606_deepfake/DeepfakeBench/training/weights/xception_best.pth",
    map_location="cpu"
)

# 구조 확인
if isinstance(state, dict):
    print("키 목록:", list(state.keys())[:10])
else:
    print("타입:", type(state))