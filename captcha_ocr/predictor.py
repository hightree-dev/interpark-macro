"""빠른 추론을 위한 ONNX 런타임 기반 CAPTCHA 예측 모듈."""

import os
import time
from typing import Tuple, Optional, Union

import numpy as np
import onnxruntime as ort

from .config import ModelConfig, DEFAULT_CONFIG
from .preprocessor import ImagePreprocessor


class CaptchaPredictor:
    """
    ONNX 런타임을 사용한 고성능 CAPTCHA 인식 클래스.
    
    주요 기능:
        - ONNX 최적화 추론
        - 가변 길이 텍스트를 위한 CTC 디코딩
        - 다양한 입력 포맷 지원 (바이트, 경로, 넘파이)
    
    사용 예시:
        >>> predictor = CaptchaPredictor("weights/captcha_model.onnx")
        >>> text = predictor.predict(image_bytes)
        >>> print(f"Predicted: {text}")
    """
    
    def __init__(
        self,
        model_path: str = None,
        config: ModelConfig = None
    ):
        """
        CAPTCHA 예측기를 초기화합니다.
        
        Args:
            model_path: ONNX 모델 파일 경로
            config: 모델 설정
        """
        self.config = config or DEFAULT_CONFIG
        
        # 모델 경로 설정 (프로젝트 루트 기준 절대 경로로 변환)
        if model_path:
            self.model_path = model_path
        else:
            # captcha_ocr/predictor.py 위치 기준으로 상위 프로젝트 루트의 weights 폴더 참조
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, self.config.model_path)
            
        self.preprocessor = ImagePreprocessor(self.config)
        self.session: Optional[ort.InferenceSession] = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """ONNX 모델을 세션에 로드합니다."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 호환성을 위해 CPU 실행 프로바이더 사용
        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def _ctc_decode(self, predictions: np.ndarray) -> str:
        """
        Greedy 디코딩을 사용하여 CTC 출력을 텍스트로 변환합니다.
        
        Args:
            predictions: (time_steps, batch, num_classes) 형태의 모델 출력
            
        Returns:
            디코딩된 텍스트 문자열
        """
        # Argmax 예측 인덱스 추출
        pred_indices = np.argmax(predictions, axis=-1)
        
        # 출력 형태 처리
        if len(pred_indices.shape) == 2:
            pred_indices = pred_indices[0]  # 첫 번째 배치 아이템 사용
        
        # 중복 제거 및 공백 토큰 제거 디코딩
        decoded = []
        prev_idx = -1
        
        for idx in pred_indices:
            if idx != prev_idx and idx != 0:  # 0은 공백(blank) 토큰
                char = self.config.idx_to_char.get(idx, '')
                if char:
                    decoded.append(char)
            prev_idx = idx
        
        return ''.join(decoded)[:self.config.max_length]
    
    def predict(
        self,
        image: Union[bytes, str, np.ndarray],
        measure_time: bool = False
    ) -> str:
        """
        이미지에서 CAPTCHA 텍스트를 예측합니다.
        
        Args:
            image: 바이트, 파일 경로 또는 넘파이 배열 형태의 이미지
            measure_time: True일 경우 추론 시간 출력
            
        Returns:
            예측된 CAPTCHA 텍스트
        """
        start_time = time.time() if measure_time else None
        
        # 이미지 전처리
        img_input = self.preprocessor.preprocess(image)
        
        # 추론 실행
        predictions = self.session.run(
            [self.output_name],
            {self.input_name: img_input}
        )[0]
        
        # 결과 디코딩
        text = self._ctc_decode(predictions)
        
        if measure_time:
            elapsed = time.time() - start_time
            print(f"⏱️ 추론 시간: {elapsed:.4f}s")
        
        return text
    
    def predict_with_confidence(
        self,
        image: Union[bytes, str, np.ndarray]
    ) -> Tuple[str, float]:
        """
        신뢰도 점수와 함께 CAPTCHA 텍스트를 예측합니다.
        
        Args:
            image: 바이트, 파일 경로 또는 넘파이 배열 형태의 이미지
            
        Returns:
            (예측 텍스트, 신뢰도 점수) 튜플
        """
        # 이미지 전처리
        img_input = self.preprocessor.preprocess(image)
        
        # 추론 실행
        predictions = self.session.run(
            [self.output_name],
            {self.input_name: img_input}
        )[0]
        
        # Softmax 적용하여 확률 계산
        probs = self._softmax(predictions)
        
        # 최대 확률의 평균으로 신뢰도 계산
        max_probs = np.max(probs, axis=-1)
        confidence = float(np.mean(max_probs))
        
        # 결과 디코딩
        text = self._ctc_decode(predictions)
        
        return text, confidence
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """입력 배열에 Softmax를 적용합니다."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predictor.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "weights/captcha_model.onnx"
    
    try:
        predictor = CaptchaPredictor(model_path)
        result = predictor.predict(image_path, measure_time=True)
        print(f"🧠 Prediction: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
