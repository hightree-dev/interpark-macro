"""CAPTCHA OCR을 위한 모델 설정 모듈."""

from dataclasses import dataclass
from typing import List
import string


@dataclass
class ModelConfig:
    """OCR 모델 학습 및 추론을 위한 설정 클래스."""
    
    # 이미지 크기
    img_width: int = 210
    img_height: int = 70
    
    # 문자 셋 (영어 대문자 전용)
    characters: List[str] = None
    max_length: int = 6
    
    # 학습 파라미터
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-3
    
    # 파일 경로
    model_path: str = "weights/captcha_model.onnx"
    weights_path: str = "weights/captcha_weights.h5"
    
    def __post_init__(self):
        if self.characters is None:
            self.characters = sorted(list(string.ascii_uppercase))
    
    @property
    def num_classes(self) -> int:
        """CTC를 위한 공백(blank) 토큰을 포함한 클래스 개수."""
        return len(self.characters) + 1
    
    @property
    def char_to_idx(self) -> dict:
        """문자를 인덱스로 매핑 (1부터 시작, 0은 공백)."""
        return {c: i + 1 for i, c in enumerate(self.characters)}
    
    @property
    def idx_to_char(self) -> dict:
        """인덱스를 문자로 매핑."""
        return {i + 1: c for i, c in enumerate(self.characters)}


# 기본 설정 인스턴스
DEFAULT_CONFIG = ModelConfig()
