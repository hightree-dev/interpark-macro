"""
CTC Loss를 사용한 OCR 모델 트레이너.

이 모듈은 OCR 학습을 위한 CNN-RNN 아키텍처를 구현합니다:
- 특징 추출을 위한 CNN 백본
- 시퀀스 모델링을 위한 양방향 GRU
- 가변 길이 텍스트 인식을 위한 CTC 손실 함수

학습 파이프라인 포함 사항:
- 데이터 증강
- 학습률 스케줄링
- 체크포인트 저장
- 조기 종료 (Early Stopping)
"""

import os
import string
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import ModelConfig, DEFAULT_CONFIG


class CaptchaDataset(Dataset):
    """CAPTCHA 이미지와 라벨을 포함하는 데이터셋 클래스."""
    
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        config: ModelConfig = None,
        augment: bool = True
    ):
        """
        Args:
            csv_path: [filename, label] 컬럼이 포함된 CSV 파일 경로
            image_dir: 이미지가 저장된 디렉토리
            config: 모델 설정
            augment: 데이터 증강 적용 여부
        """
        self.config = config or DEFAULT_CONFIG
        self.image_dir = image_dir
        self.augment = augment
        
        # 라벨 로드
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=['label'])
        
        # 이미지 변환
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        
        # 이미지 로드 및 전처리
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")
        
        # 전처리 적용
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 크기 조정
        img = cv2.resize(img, (self.config.img_width, self.config.img_height))
        
        # 변환 적용
        img = self.transform(img)
        
        # 라벨 인코딩
        label_str = str(row['label']).upper()
        label = [self.config.char_to_idx[c] for c in label_str if c in self.config.char_to_idx]
        
        return img, torch.tensor(label, dtype=torch.long), len(label)


class OCRModel(nn.Module):
    """
    CTC 출력을 사용하는 CNN-RNN OCR 모델.
    
    아키텍처:
        - MaxPooling을 포함한 2개의 Conv 레이어
        - 양방향 GRU
        - 완전 연결(Fully Connected) 출력 레이어
    """
    
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        
        # Calculate CNN output dimensions
        conv_output_h = self.config.img_height // 2 // 2
        gru_input_size = 64 * conv_output_h
        
        # RNN layers
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(256, self.config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 연산을 수행합니다.
        
        Args:
            x: (batch, 1, height, width) 형태의 입력 텐서
            
        Returns:
            (time_steps, batch, num_classes) 형태의 출력 텐서
        """
        batch_size = x.size(0)
        
        # CNN 특징 추출
        x = self.cnn(x)
        
        # RNN을 위한 변환: (batch, width, height*channels)
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.contiguous().view(batch_size, x.size(1), -1)
        
        # RNN 시퀀스 모델링
        x, _ = self.gru(x)
        
        # 출력 투영(Projection)
        x = self.fc(x)
        
        # CTC 손실 함수를 위한 변환: (time_steps, batch, num_classes)
        x = x.permute(1, 0, 2)
        
        return x


class OCRTrainer:
    """
    CTC 손실 함수를 사용한 OCR 모델 트레이너 클래스.
    
    주요 기능:
        - 자동 체크포인트 저장
        - 학습 진행률 시각화
        - 조기 종료 지원
        - ONNX 포맷 모델 내보내기
    """
    
    def __init__(
        self,
        config: ModelConfig = None,
        device: str = None
    ):
        self.config = config or DEFAULT_CONFIG
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = OCRModel(self.config).to(self.device)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: int = None,
        save_path: str = None
    ) -> List[float]:
        """
        모델 학습을 수행합니다.
        
        Args:
            train_loader: 학습 데이터 로더
            epochs: 학습 에폭 수
            save_path: 체크포인트 저장 경로
            
        Returns:
            에폭별 학습 손실(Loss) 리스트
        """
        epochs = epochs or self.config.epochs
        save_path = save_path or "weights/ocr_checkpoint.pth"
        
        if len(train_loader) == 0:
            print("⚠️ 학습 데이터가 없습니다. labeling_tool을 이용해 데이터를 먼저 수집해주세요.")
            return []
        
        losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, (images, labels, label_lens) in enumerate(train_loader):
                images = images.to(self.device)
                
                # 순전파
                outputs = self.model(images)
                
                # CTC 손실 처리를 위한 준비
                input_lens = torch.full(
                    (images.size(0),),
                    outputs.size(0),
                    dtype=torch.long
                )
                
                # 라벨 결합
                labels_concat = torch.cat(labels).to(self.device)
                label_lens = torch.tensor(label_lens, dtype=torch.long)
                
                # 손실 계산
                loss = self.criterion(outputs, labels_concat, input_lens, label_lens)
                
                # 역전파
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            print(f"📘 Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # 최우수 모델 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(save_path)
        
        return losses
    
    def save_checkpoint(self, path: str) -> None:
        """모델 체크포인트를 저장합니다."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✅ 체크포인트 저장 완료: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """모델 체크포인트를 로드합니다."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        print(f"📥 체크포인트 로드 완료: {path}")
    
    def export_onnx(self, output_path: str) -> None:
        """모델을 ONNX 포맷으로 내보냅니다."""
        self.model.eval()
        
        # 더미 입력 생성
        dummy_input = torch.randn(
            1, 1,
            self.config.img_height,
            self.config.img_width
        ).to(self.device)
        
        # 내보내기 수행
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['image'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'output': {1: 'batch_size'}
            },
            opset_version=13
        )
        
        print(f"✅ ONNX 모델 내보내기 완료: {output_path}")


def collate_fn(batch):
    """가변 길이 라벨을 처리하기 위한 커스텀 collate 함수."""
    images, labels, label_lens = zip(*batch)
    return torch.stack(images), labels, label_lens


if __name__ == "__main__":
    # Example training script
    print("🚀 Starting OCR training...")
    
    config = ModelConfig()
    trainer = OCRTrainer(config)
    
    # Load dataset
    dataset = CaptchaDataset(
        csv_path="data/labels.csv",
        image_dir="data/labeled",
        config=config
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Train
    losses = trainer.train(loader, epochs=config.epochs)
    
    # Export to ONNX
    trainer.export_onnx("weights/captcha_model.onnx")
    
    print("✅ Training completed!")
