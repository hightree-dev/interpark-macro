"""CAPTCHA OCR을 위한 이미지 전처리 유틸리티."""

import numpy as np
from PIL import Image
import io
from typing import Union

from .config import ModelConfig, DEFAULT_CONFIG


class ImagePreprocessor:
    """OCR 모델 추론을 위한 이미지 전처리 클래스."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or DEFAULT_CONFIG
    
    def preprocess(self, image: Union[bytes, str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        모델 입력을 위한 이미지 전처리를 수행합니다.
        
        Args:
            image: 바이트, 파일 경로, 넘파이 배열 또는 PIL 이미지 형태의 이미지
            
        Returns:
            (1, width, height, 1) 형태의 전처리된 이미지 배열
        """
        # PIL 이미지로 변환
        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image)).convert('L')
        elif isinstance(image, str):
            img = Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('L')
        elif isinstance(image, Image.Image):
            img = image.convert('L')
        else:
            raise ValueError(f"지원되지 않는 이미지 타입입니다: {type(image)}")
        
        # 크기 조정
        img = img.resize((self.config.img_width, self.config.img_height))
        
        # 넘파이 배열로 변환 및 정규화
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 전치 (H, W) → (W, H) - CTC 입력 사양에 맞춤
        img_array = np.transpose(img_array)
        
        # 배치 및 채널 차원 추가
        img_array = img_array.reshape(1, self.config.img_width, self.config.img_height, 1)
        
        return img_array
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """이미지 배치를 전처리합니다."""
        processed = [self.preprocess(img) for img in images]
        return np.concatenate(processed, axis=0)
