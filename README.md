# DesignBy
Home appliance design classification model


#### Structure
```bash
CNN/
│
├── models/
│   └── resnet.py         # ResNet18 모델 정의 파일
│
├── utils.py              # 데이터 로드, 전처리, DataLoader 생성 등 유틸리티 함수 제공
│
├── train.py              # 모델 학습, 검증, 평가를 담당
├── run.py                # 여러 질문에 대해 학습을 반복 실행 (진입점)
├── config.json           # 구성 파일 (질문 목록, 경로, 배치 크기 등 설정)
├── logs/                 # 학습 및 검증 로그 저장 디렉토리
└── trained_models/       # 질문별로 저장된 학습 완료 모델 디렉토리
``` 

#### Training for 22 questions
```bash
python main.py
```

#### Training for 1 questions
```bash
python run.py
```
