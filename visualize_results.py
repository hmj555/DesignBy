import matplotlib.pyplot as plt
import pandas as pd

def plot_training_log(log_path, output_path):
    """
    학습 로그 데이터를 시각화.
    
    Args:
        log_path (str): 학습 로그 파일 경로 (CSV 형식).
        output_path (str): 그래프 저장 경로.
    """
    # 로그 파일 읽기
    data = pd.read_csv(log_path)
    
    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data['train_loss'], label='Train Loss', marker='o')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', marker='o')
    plt.plot(data['epoch'], data['accuracy'], label='Validation Accuracy', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and Validation Results")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    print(f"Training graph saved to {output_path}")
