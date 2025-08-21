import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from einops import rearrange
from inspect import isfunction
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ======================== 모델 구성 요소 ========================

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.2):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))
        
    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.3):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
    
    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.3, context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=3, dropout=0.3, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for _ in range(depth)])
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

def nonlinearity(x):
    return x * torch.sigmoid(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        
    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, downsample_steps, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=1, padding=2)])
        for _ in range(downsample_steps):
            self.layers.append(nn.Sequential(ResnetBlock(hidden_channels, hidden_channels, dropout), Downsample(hidden_channels)))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, upsample_steps, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(upsample_steps):
            self.layers.append(nn.Sequential(ResnetBlock(hidden_channels, hidden_channels, dropout), Upsample(hidden_channels)))
        self.norm = Normalize(hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return torch.tanh(x)

def build_grid(resolution, max_v=1.):
    ranges = [np.linspace(0., max_v, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, max_v - grid], axis=-1)

class PosEmbeds(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.linear = nn.Linear(4, hidden_size)
        self.grid = nn.Parameter(torch.Tensor(build_grid(resolution)), requires_grad=False)
        
    def forward(self, inputs):
        pos_emb = self.linear(self.grid).moveaxis(3, 1)
        return inputs + pos_emb

# ======================== TransPath 모델 ========================

class TransPathPPM(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, hidden_channels=64, attn_blocks=4, attn_heads=4, 
                 cnn_dropout=0.15, attn_dropout=0.15, downsample_steps=3, resolution=(64, 64)):
        super().__init__()
        heads_dim = hidden_channels // attn_heads
        self.encoder = Encoder(in_channels, hidden_channels, downsample_steps, cnn_dropout)
        self.pos = PosEmbeds(hidden_channels, (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps))
        self.transformer = SpatialTransformer(hidden_channels, attn_heads, heads_dim, attn_blocks, attn_dropout)
        self.decoder_pos = PosEmbeds(hidden_channels, (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps))
        self.decoder = Decoder(hidden_channels, out_channels, downsample_steps, cnn_dropout)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.pos(x)
        x = self.transformer(x)
        x = self.decoder_pos(x)
        x = self.decoder(x)
        return x

# ======================== 데이터셋 클래스 ========================

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

# ======================== 학습 함수 ========================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        outputs = (outputs + 1) / 2  # tanh 출력을 0-1로 정규화
        
        # Loss 계산
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계 업데이트
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            outputs = (outputs + 1) / 2
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

# ======================== 메인 학습 코드 ========================

def main():
    # 설정
    DATA_ROOT = "TransPath_data"  # 데이터 경로
    BATCH_SIZE = 32  # 배치 크기 (메모리에 따라 조정)
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    SUBSET_RATIO = 1.0  # 전체 데이터의 10%만 사용 (테스트용, 전체 사용시 1.0)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # 데이터셋 생성
    print("데이터셋 로딩 중...")
    train_dataset = NPYDataset(DATA_ROOT, split='train', subset_ratio=SUBSET_RATIO)
    val_dataset = NPYDataset(DATA_ROOT, split='val', subset_ratio=SUBSET_RATIO)
    
    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 모델 생성
    print("모델 생성 중...")
    model = TransPathPPM(
        in_channels=2,
        out_channels=1,
        hidden_channels=64,
        attn_blocks=4,
        attn_heads=4,
        resolution=(64, 64)
    ).to(DEVICE)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 옵티마이저와 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    
    # 학습 기록
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 학습 루프
    print("\n학습 시작...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        
        # 학습률 조정
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 베스트 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_ppm_model100.pth')
            print(f"Best model saved! (Val Loss: {best_val_loss:.4f})")
    
    print("\n학습 완료!")
    
    # 학습 곡선 그리기
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    # 예측 시각화 (마지막 배치)
    plt.subplot(1, 2, 2)
    model.eval()
    with torch.no_grad():
        sample_inputs, sample_targets = next(iter(val_loader))
        sample_inputs = sample_inputs[:4].to(DEVICE)  # 4개 샘플만
        sample_targets = sample_targets[:4].to(DEVICE)
        sample_outputs = model(sample_inputs)
        sample_outputs = (sample_outputs + 1) / 2
        
        # 첫 번째 샘플 시각화
        plt.subplot(1, 4, 1)
        plt.imshow(sample_inputs[0, 0].cpu(), cmap='gray')
        plt.title('Map')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(sample_inputs[0, 1].cpu(), cmap='hot')
        plt.title('Start+Goal')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(sample_targets[0, 0].cpu(), cmap='hot')
        plt.title('GT PPM')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(sample_outputs[0, 0].cpu(), cmap='hot')
        plt.title('Pred PPM')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    return model

# ======================== CSV 맵 적용 함수 ========================

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

# ======================== 실행 ========================

if __name__ == "__main__":
    # 학습 실행
    trained_model = main()
    
    print("\n모델 학습이 완료되었습니다!")
    print("저장된 모델: best_ppm_model.pth")
    print("\nCSV 맵에 적용하려면:")
    print("1. 모델 로드: checkpoint = torch.load('best_ppm_model.pth')")
    print("2. model.load_state_dict(checkpoint['model_state_dict'])")
    print("3. predicted_ppm = apply_to_csv_map(model, csv_map, start, goal)")