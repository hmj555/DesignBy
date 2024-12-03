### run.py
### 특정 질문에 대한 학습 실행

from train import train_model
from visualize_results import plot_training_log
from utils import format_filename, get_dataloader
from models.resnet import create_resnet18
import os
import torch
from train import test_model

# 특정 질문에 대한 실행
question = "what is the score for the straight style?"
batch_size = 16
epochs = 10
base_path = "/data/gist/DesignBy_split_"
log_dir = "logs"
model_dir = "trained_models"
device = "cuda"

# 경로 설정
filename = format_filename(question)
log_path = os.path.join(log_dir, filename, "train.log")
model_path = os.path.join(model_dir, filename, "model.pt")
graph_path = os.path.join(log_dir, filename, "training_graph.png")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# 모델 학습
train_model(
    question=question,
    epochs=epochs,
    batch_size=batch_size,
    model_path=model_path,
    log_path=log_path,
    device=device,
    base_path=base_path
)

# 테스트 평가
_, _, test_loader = get_dataloader(question, batch_size, base_path)
model = create_resnet18(num_classes=6).to(device)
model.load_state_dict(torch.load(model_path))
accuracy = test_model(model, test_loader, device)
print(f"Test Accuracy for '{question}': {accuracy:.4f}")

# 결과 시각화
plot_training_log(log_path, graph_path)
print(f"Training log saved at {graph_path}")
