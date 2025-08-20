import torch
import numpy as np
from model import TransPathPPM

def apply_to_csv_map(model, csv_map, start_pos, goal_pos, device='cuda'):
    """
    학습된 모델을 CSV 맵에 적용
    
    Args:
        model: 학습된 TransPathPPM 모델
        csv_map: CSV에서 로드한 맵 (1=길, 0=장애물)
        start_pos: (y, x) 시작 위치
        goal_pos: (y, x) 목표 위치
    
    Returns:
        predicted_ppm: 예측된 PPM
    """
    model.eval()
    
    # CSV 맵 인코딩 변환 (CSV와 NPY가 반대)
    npy_format_map = 1 - csv_map  # 0=길, 1=장애물로 변환
    
    # 입력 데이터 준비
    map_channel = torch.tensor(npy_format_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
    
    # Start+Goal 채널 생성
    query_channel = torch.zeros_like(map_channel)
    query_channel[0, 0, start_pos[0], start_pos[1]] = 1.0
    query_channel[0, 0, goal_pos[0], goal_pos[1]] = 1.0
    
    # 2채널 입력 구성
    inputs = torch.cat([map_channel, query_channel], dim=1).to(device)  # (1, 2, 64, 64)
    
    # 예측
    with torch.no_grad():
        outputs = model(inputs)
        outputs = (outputs + 1) / 2  # tanh를 0-1로 정규화
    
    return outputs.squeeze().cpu().numpy()

def load_trained_model(model_path, device='cuda'):
    """
    학습된 모델 로드
    
    Args:
        model_path: 모델 파일 경로
        device: 사용할 디바이스
    
    Returns:
        model: 로드된 모델
    """
    # 모델 생성
    model = TransPathPPM(
        in_channels=2,
        out_channels=1,
        hidden_channels=64,
        attn_blocks=4,
        attn_heads=4,
        resolution=(64, 64)
    ).to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"모델 로드 완료! (Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")
    
    return model

def predict_path_probabilities(model_path, csv_map, start_pos, goal_pos, device='cuda'):
    """
    완전한 추론 파이프라인
    
    Args:
        model_path: 학습된 모델 파일 경로
        csv_map: CSV 맵 데이터
        start_pos: 시작 위치 (y, x)
        goal_pos: 목표 위치 (y, x)
        device: 사용할 디바이스
    
    Returns:
        predicted_ppm: 예측된 경로 확률 맵
    """
    # 모델 로드
    model = load_trained_model(model_path, device)
    
    # 예측 수행
    predicted_ppm = apply_to_csv_map(model, csv_map, start_pos, goal_pos, device)
    
    return predicted_ppm