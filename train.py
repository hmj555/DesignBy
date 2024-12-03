### train.py
### 모델 학습 및 평가

import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet import create_resnet18
from utils import get_dataloader


def train_model(question, epochs, batch_size, model_path, log_path, device="cuda", base_path=None):
    """
    ResNet18 모델 학습 함수

    Args:
        question (str): 학습할 질문 텍스트
        epochs (int): 총 학습 에폭 수
        batch_size (int): 배치 크기
        model_path (str): 모델 저장 경로
        log_path (str): 로그 저장 경로
        device (str): 사용할 장치 ('cuda' 또는 'cpu')
         base_path (str): 데이터 경로

    Returns:
        None
    """
    train_loader, val_loader, _ = get_dataloader(question, batch_size, base_path)
    model = create_resnet18(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 로그 파일 준비
    with open(log_path, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss,accuracy\n")  # CSV 형식 로그

    for epoch in range(epochs):
        # 학습 루프
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # 검증 루프
        val_loss, accuracy = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 로그 저장
        with open(log_path, "a") as log_file:
            log_file.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f},{accuracy:.4f}\n")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def evaluate(model, val_loader, criterion, device):
    """
    검증 루프
    Args:
        model: 학습 중인 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: 학습 장치

    Returns:
        avg_loss: 검증 손실의 평균
        accuracy: 검증 정확도
    """
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def test_model(model, test_loader, device="cuda"):
    """
    테스트 데이터 평가.
    Args:
        model: 학습된 모델
        test_loader: 테스트 데이터 로더
        device: 학습 장치

    Returns:
        accuracy: 테스트 정확도
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy
