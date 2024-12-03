import os
import json
import csv
import torch
from run import train_model
from visualize_results import plot_training_log
from utils import get_dataloader, format_filename
from models.resnet import create_resnet18
from train import test_model

def main():
    # Config 파일 로드
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    base_path = config["base_path"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    questions = config["questions"]
    log_dir = config["log_dir"]
    model_dir = config["model_dir"]
    device = config["device"]
    summary_path = "results_summary.csv"

    # 결과 요약 CSV 초기화
    with open(summary_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Question", "Test Accuracy", "Log Path", "Model Path", "Graph Path"])

    # 학습 루프
    for question in questions:
        print(f"Training for '{question}'")
        filename = format_filename(question)
        question_log_dir = os.path.join(log_dir, filename)
        question_model_dir = os.path.join(model_dir, filename)
        os.makedirs(question_log_dir, exist_ok=True)
        os.makedirs(question_model_dir, exist_ok=True)

        log_path = os.path.join(question_log_dir, "train.log")
        model_path = os.path.join(question_model_dir, "model.pt")
        graph_path = os.path.join(question_log_dir, "training_graph.png")

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
        test_accuracy = test_model(model, test_loader, device)

        # 결과 시각화
        plot_training_log(log_path, graph_path)

        # 결과 저장
        with open(summary_path, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([question, test_accuracy, log_path, model_path, graph_path])

    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
