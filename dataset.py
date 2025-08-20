import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class NPYDataset(Dataset):
    def __init__(self, data_dir, split='train', clip_value=0.95, subset_ratio=1.0):
        """
        NPY 데이터셋 로더
        Args:
            data_dir: TransPath_data 폴더 경로
            split: 'train', 'val', 'test' 중 하나
            clip_value: focal 값 필터링 임계값
            subset_ratio: 데이터 사용 비율 (메모리 절약용)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.clip_value = clip_value
        
        # 데이터 파일 경로
        split_dir = self.data_dir / split
        
        # 메모리 매핑 모드로 로드 (메모리 절약)
        self.maps = np.load(split_dir / 'maps.npy', mmap_mode='r')
        self.starts = np.load(split_dir / 'starts.npy', mmap_mode='r')
        self.goals = np.load(split_dir / 'goals.npy', mmap_mode='r')
        self.focal = np.load(split_dir / 'focal.npy', mmap_mode='r')
        
        # 데이터 서브셋 사용
        total_size = len(self.maps)
        self.size = int(total_size * subset_ratio)
        
        print(f"{split} 데이터셋 로드 완료: {self.size}/{total_size} 샘플")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 데이터 로드
        map_data = self.maps[idx].astype(np.float32)  # (1, 64, 64)
        start_data = self.starts[idx].astype(np.float32)  # (1, 64, 64)
        goal_data = self.goals[idx].astype(np.float32)  # (1, 64, 64)
        focal_data = self.focal[idx].astype(np.float32)  # (1, 64, 64)
        
        # 입력 데이터 구성: 2채널 (maps, starts+goals)
        query_channel = start_data + goal_data  # 시작점과 목표점 합치기
        inputs = np.concatenate([map_data, query_channel], axis=0)  # (2, 64, 64)
        
        # Focal 값 필터링 (0.95 미만은 0으로)
        focal_filtered = np.where(focal_data >= self.clip_value, focal_data, 0.0)
        
        # 텐서로 변환
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(focal_filtered)
        
        return inputs, targets