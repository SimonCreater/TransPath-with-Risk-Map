import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def visualize_results(model, val_loader, device, save_path='training_results.png'):
    """학습 결과 시각화"""
    model.eval()
    
    with torch.no_grad():
        sample_inputs, sample_targets = next(iter(val_loader))
        sample_inputs = sample_inputs[:4].to(device)  # 4개 샘플만
        sample_targets = sample_targets[:4].to(device)
        sample_outputs = model(sample_inputs)
        sample_outputs = (sample_outputs + 1) / 2
        
        # 첫 번째 샘플 시각화
        plt.figure(figsize=(16, 4))
        
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
    plt.savefig(save_path)
    plt.show()

def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """학습 곡선 그리기"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()