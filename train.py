import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from model import TransPathPPM
from dataset import NPYDataset
from training_utils import train_epoch, validate, visualize_results, plot_training_history

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
            }, 'best_ppm_model.pth')
            print(f"Best model saved! (Val Loss: {best_val_loss:.4f})")
    
    print("\n학습 완료!")
    
    # 결과 시각화
    plot_training_history(train_losses, val_losses)
    visualize_results(model, val_loader, DEVICE)
    
    return model

if __name__ == "__main__":
    trained_model = main()
    
    print("\n모델 학습이 완료되었습니다!")
    print("저장된 모델: best_ppm_model.pth")
    print("\nCSV 맵에 적용하려면:")
    print("1. 모델 로드: checkpoint = torch.load('best_ppm_model.pth')")
    print("2. model.load_state_dict(checkpoint['model_state_dict'])")
    print("3. predicted_ppm = apply_to_csv_map(model, csv_map, start, goal)")