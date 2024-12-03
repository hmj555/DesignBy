### utils.py
### 데이터셋 로드, DataLoader 생성, 모델 학습을 위한 유틸리티 함수

import os
import json ; import re 
from PIL import Image ; import pillow_avif
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# 데이터셋 경로
#/data/gist/DesignBy_split_
#### / train
######## / imgaes
######## / custom_question_train.json
######## / custom_annotation_train.json
#### / val
######## / imgaes
######## / custom_question_val.json
######## / custom_annotation_val.json
#### / test
######## / imgaes
######## / custom_question_test.json
######## / custom_annotation_test.json
#####################

# 파일명 포맷팅
def format_filename(question):
    """
    Args:
        question (str): 질문 텍스트 (e.g., "what is the score for the straight style?")
    
    Returns:
        str: 간결한 파일명 (e.g., "straight_style")
    """
    keywords = re.sub(r"^what is the score for the ", "", question, flags=re.IGNORECASE)
    formatted_name = "_".join(keywords.strip().replace("?", "").split()).lower()
    return formatted_name



class CustomDataset(Dataset):
    def __init__(self, image_dir, question_file, annotation_file, question_text, transform=None):
        """
        image_dir: 이미지 디렉토리 경로
        question_file: 질문 JSON 파일 경로
        annotation_file: 주석 JSON 파일 경로
        question_text: 분석할 질문 텍스트
        transform: 이미지 전처리 변환
        """
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.data = self._load_data(question_file, annotation_file, question_text)

    def _load_data(self, question_file, annotation_file, question_text):
        # 질문 및 주석 데이터 로드
        with open(question_file, 'r') as qf:
            questions = json.load(qf)['questions']
        with open(annotation_file, 'r') as af:
            annotations = json.load(af)['annotations']

        # 데이터 필터링 및 매핑
        data = []
        for question in questions:
            if question['question'] == question_text:
                question_id = question['question_id']
                image_id = question['image_id']
                annotation = next((a for a in annotations if a['question_id'] == question_id), None)
                if annotation:
                    label = int(annotation['multiple_choice_answer'])
                    image_file = self._find_image_file(image_id)
                    if image_file:
                        data.append((image_file, label))
        return data

    def _find_image_file(self, image_id):
        """
        이미지 파일을 찾아 확장자를 처리하는 메서드
        """
        for filename in os.listdir(self.image_dir):
            if filename.startswith(image_id):
                file_path = os.path.join(self.image_dir, filename)
                
                # avif 확장자 처리
                if filename.endswith('.avif'):
                    png_path = file_path.replace('.avif', '.png')
                    if not os.path.exists(png_path):  # 이미 변환된 파일이 없다면 변환
                        self._convert_avif_to_png(file_path, png_path)
                    return png_path  # 변환된 PNG 경로 반환
                
                # jfif 확장자 처리
                if filename.endswith('.jfif'):
                    png_path = file_path.replace('.jfif', '.png')
                    if not os.path.exists(png_path):  # 이미 변환된 파일이 없다면 변환
                        self._convert_jfif_to_png(file_path, png_path)
                    return png_path  # 변환된 PNG 경로 반환
                
                # 기타 이미지 파일 (e.g., jpg, jpeg, png 등)
                return file_path
        return None

    @staticmethod
    def _convert_avif_to_png(avif_path, png_path):
        # AVIF -> PNG 변환
        try:
            with Image.open(avif_path) as img:
                img = img.convert("RGB")
                img.save(png_path, "PNG")
            print(f"Converted {avif_path} to {png_path}")
        except Exception as e:
            print(f"Error converting {avif_path} to PNG: {e}")

    @staticmethod
    def _convert_jfif_to_png(jfif_path, png_path):
        """
        JFIF -> PNG 변환
        """
        try:
            with Image.open(jfif_path) as img:
                img = img.convert("RGB")  # RGB 모드로 변환
                img.save(png_path, "PNG")
            print(f"Converted {jfif_path} to {png_path}")
        except Exception as e:
            print(f"Error converting {jfif_path} to PNG: {e}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloader(split_path, question_text, batch_size, shuffle):
    """
    DataLoader 생성 함수
    split_path: train/val/test 경로
    question_text: 질문 텍스트 (e.g., "what is the score for the modern style?")
    batch_size: DataLoader 배치 크기
    shuffle: 데이터 셔플 여부
    """
    image_dir = os.path.join(split_path, "images")
    question_file = os.path.join(split_path, f"custom_question_{os.path.basename(split_path)}.json")
    annotation_file = os.path.join(split_path, f"custom_annotation_{os.path.basename(split_path)}.json")

    dataset = CustomDataset(image_dir, question_file, annotation_file, question_text)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_data(base_path, question_text, batch_size=32):
    """
    데이터셋 로드 함수
    base_path: 데이터셋 루트 경로
    question_text: 분석할 질문 텍스트
    batch_size: DataLoader 배치 크기
    """
    train_loader = create_dataloader(os.path.join(base_path, "train"), question_text, batch_size, shuffle=True)
    val_loader = create_dataloader(os.path.join(base_path, "val"), question_text, batch_size, shuffle=False)
    test_loader = create_dataloader(os.path.join(base_path, "test"), question_text, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_dataloader(question, batch_size, base_path):
    train_loader = create_dataloader(os.path.join(base_path, "train"), question, batch_size, shuffle=True)
    val_loader = create_dataloader(os.path.join(base_path, "val"), question, batch_size, shuffle=False)
    test_loader = create_dataloader(os.path.join(base_path, "test"), question, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader



################## 테스트 코드

if __name__ == "__main__":
    base_path = "/data/gist/DesignBy_split_"
    question_text = "what is the score for the simple style?"
    batch_size = 16

    # 데이터 로드
    train_loader, val_loader, test_loader = load_data(base_path, question_text, batch_size)

    # 첫 번째 배치 확인
    for images, labels in train_loader:
        print("Batch images shape:", images.shape)  # [batch_size, 3, 224, 224]
        print("Batch labels:", labels)             # tensor([0, 1, 2, ...])
        break
