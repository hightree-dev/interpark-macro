"""
자동 티켓 예매를 위한 CAPTCHA 핸들러.

브라우저 자동화와 통합하여 다음을 수행합니다:
1. CAPTCHA 이미지 감지
2. 이미지 데이터 추출
3. OCR 예측 실행
4. 결과 자동 입력
"""

import os
import base64
import time
from typing import Optional

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from captcha_ocr import CaptchaPredictor


class CaptchaHandler:
    """
    티켓 예매 과정 중 발생하는 CAPTCHA 챌린지를 처리합니다.
    
    이 클래스는 OCR 예측 모델을 브라우저 자동화와 통합하여
    CAPTCHA를 자동으로 해결합니다.
    
    사용 예시:
        >>> handler = CaptchaHandler(driver)
        >>> success = handler.solve_captcha()
    """
    
    # 인터파크 전용 CSS 선택자
    CAPTCHA_IMAGE_SELECTOR = "#imgCaptcha"
    CAPTCHA_INPUT_SELECTOR = "#inputCaptcha"
    CAPTCHA_REFRESH_SELECTOR = ".refreshBtn"
    
    def __init__(
        self,
        driver: WebDriver,
        predictor: CaptchaPredictor = None,
        model_path: str = "weights/captcha_model.onnx"
    ):
        """
        CAPTCHA 핸들러를 초기화합니다.
        
        Args:
            driver: 셀레니움 WebDriver 인스턴스
            predictor: 선택적으로 이미 초기화된 예측기
            model_path: ONNX 모델 파일 경로
        """
        self.driver = driver
        self.predictor = predictor
        self.model_path = model_path
        
        # 예측기 지연 로딩 (Lazy Loading)
        if self.predictor is None and os.path.exists(model_path):
            self._init_predictor()
    
    def _init_predictor(self) -> None:
        """CAPTCHA 예측기를 초기화합니다."""
        try:
            self.predictor = CaptchaPredictor(self.model_path)
            print("✅ CAPTCHA 예측기가 초기화되었습니다")
        except Exception as e:
            print(f"⚠️ 예측기 초기화 실패: {e}")
            self.predictor = None
    
    def wait_for_captcha(self, timeout: int = 10) -> Optional[WebElement]:
        """
        CAPTCHA 이미지가 나타날 때까지 대기합니다.
        
        Args:
            timeout: 최대 대기 시간 (초)
            
        Returns:
            CAPTCHA 이미지 엘리먼트 또는 None
        """
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.CAPTCHA_IMAGE_SELECTOR))
            )
            # 이미지가 완전히 로드될 때까지 잠시 대기
            time.sleep(0.1)
            return element
        except Exception as e:
            print(f"⚠️ CAPTCHA를 찾을 수 없습니다: {e}")
            return None
    
    def extract_captcha_image(self, element: WebElement = None) -> Optional[bytes]:
        """
        페이지에서 CAPTCHA 이미지 데이터를 추출합니다.
        
        Args:
            element: 선택적인 CAPTCHA 이미지 엘리먼트
            
        Returns:
            이미지 바이트 데이터 또는 None
        """
        if element is None:
            element = self.wait_for_captcha()
        
        if element is None:
            return None
        
        try:
            # 이미지 소스(base64 데이터 URL) 가져오기
            img_src = element.get_attribute('src')
            
            if not img_src:
                return None
            
            # base64 데이터 추출
            if ',' in img_src:
                base64_data = img_src.split(',')[1]
            else:
                base64_data = img_src
            
            return base64.b64decode(base64_data)
            
        except Exception as e:
            print(f"❌ CAPTCHA 이미지 추출 실패: {e}")
            return None
    
    def predict_captcha(self, image_bytes: bytes) -> Optional[str]:
        """
        이미지 바이트에서 CAPTCHA 텍스트를 예측합니다.
        
        Args:
            image_bytes: CAPTCHA 이미지 데이터
            
        Returns:
            예측된 텍스트 또는 None
        """
        if self.predictor is None:
            print("❌ 예측기가 초기화되지 않았습니다")
            return None
        
        try:
            text = self.predictor.predict(image_bytes, measure_time=True)
            print(f"🧠 예측 결과: {text}")
            return text
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return None
    
    def input_captcha(self, text: str, use_clipboard: bool = True) -> bool:
        """
        폼에 CAPTCHA 텍스트를 입력합니다.
        
        Args:
            text: 입력할 CAPTCHA 텍스트
            use_clipboard: 클립보드를 통한 입력 여부 (탐지 우회용)
            
        Returns:
            성공 여부
        """
        try:
            if use_clipboard:
                # 클립보드 기반 입력을 위한 패키지 (pyperclip, pyautogui)
                import pyperclip
                import pyautogui
                
                # 입력창 찾기 및 클릭
                input_element = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, self.CAPTCHA_INPUT_SELECTOR))
                )
                input_element.click()
                
                # 클립보드에서 붙여넣기
                pyperclip.copy(text)
                pyautogui.hotkey('command', 'v')
                
            else:
                # 직접 입력 (봇 탐지에 걸릴 가능성이 높음)
                input_element = self.driver.find_element(By.CSS_SELECTOR, self.CAPTCHA_INPUT_SELECTOR)
                input_element.clear()
                input_element.send_keys(text)
            
            return True
            
        except Exception as e:
            print(f"❌ CAPTCHA 입력 실패: {e}")
            return False
    
    def refresh_captcha(self) -> bool:
        """
        CAPTCHA 이미지를 새로고침합니다.
        
        Returns:
            성공 여부
        """
        try:
            refresh_btn = self.driver.find_element(By.CSS_SELECTOR, self.CAPTCHA_REFRESH_SELECTOR)
            refresh_btn.click()
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"❌ CAPTCHA 새로고침 실패: {e}")
            return False
    
    def solve_captcha(self, max_attempts: int = 3) -> bool:
        """
        CAPTCHA를 자동으로 해결하려고 시도합니다.
        
        Args:
            max_attempts: 최대 시도 횟수
            
        Returns:
            해결 성공 시 True
        """
        for attempt in range(max_attempts):
            print(f"🔄 CAPTCHA 시도 {attempt + 1}/{max_attempts}")
            
            # 대기 및 이미지 추출
            element = self.wait_for_captcha()
            if element is None:
                continue
            
            image_bytes = self.extract_captcha_image(element)
            if image_bytes is None:
                self.refresh_captcha()
                continue
            
            # 예측
            text = self.predict_captcha(image_bytes)
            if text is None or len(text) != 6:
                self.refresh_captcha()
                continue
            
            # 입력
            if self.input_captcha(text):
                print(f"✅ CAPTCHA 해결됨: {text}")
                return True
            
            self.refresh_captcha()
        
        print(f"❌ {max_attempts}번의 시도 후에도 CAPTCHA 해결에 실패했습니다")
        return False
