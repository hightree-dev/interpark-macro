"""
인터파크 CAPTCHA 이미지 수집을 위한 크롤러.

핵심 기능:
- Undetected Chromedriver를 사용한 봇 탐지 우회
- 자동 로그인 및 지정 페이지 이동
- CAPTCHA 이미지 반복 추출 및 PNG 저장
"""

import os
import base64
import time
from typing import Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.chrome.webdriver import WebDriver

try:
    import undetected_chromedriver as uc
except ImportError:
    uc = None


class CaptchaCrawler:
    """
    학습 데이터 구축을 위해 대량의 CAPTCHA 이미지를 수집하는 클래스.
    
    사용 예시:
        >>> crawler = CaptchaCrawler()
        >>> crawler.collect(count=100)
    """
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        headless: bool = False
    ):
        """
        크롤러를 초기화합니다.
        
        Args:
            output_dir: 이미지를 저장할 디렉토리
            headless: 브라우저 창 표시 여부
        """
        self.output_dir = output_dir
        self.headless = headless
        self.driver: Optional[WebDriver] = None
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _setup_driver(self) -> None:
        """Undetected Chromedriver를 설정합니다."""
        if uc is None:
            raise ImportError(
                "undetected-chromedriver가 필요합니다. "
                "설치: pip install undetected-chromedriver"
            )
        
        options = uc.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        self.driver = uc.Chrome(options=options)
        self.driver.implicitly_wait(10)
        print("✅ Chrome 드라이버가 초기화되었습니다")
    
    def login(self) -> bool:
        """인터파크에 로그인을 수행합니다 (환경변수 사용)."""
        user_id = os.getenv("INTERPARK_ID")
        user_pw = os.getenv("INTERPARK_PW")
        
        if not user_id or not user_pw:
            print("❌ 환경변수에서 ID/PW를 찾을 수 없습니다 (.env 파일을 확인하세요)")
            return False
            
        try:
            self.driver.get("https://accounts.interpark.com/login")
            
            # ID/PW 입력
            self.driver.find_element(By.ID, "userId").send_keys(user_id)
            self.driver.find_element(By.ID, "userPw").send_keys(user_pw)
            
            # 로그인 버튼 클릭
            self.driver.find_element(By.ID, "btn_login").click()
            time.sleep(2)
            
            print("✅ 로그인 성공")
            return True
        except Exception as e:
            print(f"❌ 로그인 실패: {e}")
            return False
            
    def navigate_to_target(self, url: str) -> bool:
        """CAPTCHA가 발생하는 대상 페이지로 이동합니다."""
        try:
            self.driver.get(url)
            time.sleep(2)
            print(f"📍 대상 페이지 이동: {url}")
            return True
        except Exception as e:
            print(f"❌ 페이지 이동 실패: {e}")
            return False
            
    def save_image(self, img_bytes: bytes) -> str:
        """이미지 바이트를 PNG 파일로 저장합니다."""
        # 타임스탬프 기반 고유 파일명 생성
        filename = f"{int(time.time() * 1000)}.png"
        path = os.path.join(self.output_dir, filename)
        
        with open(path, "wb") as f:
            f.write(img_bytes)
            
        return filename
        
    def collect(self, count: int = 100, target_url: str = None) -> int:
        """
        지정된 수만큼 CAPTCHA 이미지를 수집합니다.
        
        Args:
            count: 수집할 이미지 개수
            target_url: 대상 페이지 URL
            
        Returns:
            성공적으로 수집된 이미지 개수
        """
        if self.driver is None:
            self._setup_driver()
            
        if not self.login():
            return 0
            
        if target_url:
            self.navigate_to_target(target_url)
            
        collected = 0
        print(f"📦 {count}개의 CAPTCHA 이미지 수집을 시작합니다...")
        
        try:
            # CAPTCHA 핸들러 초기화 (이미지 추출용)
            from automation.captcha_handler import CaptchaHandler
            handler = CaptchaHandler(self.driver)
            
            while collected < count:
                # 이미지 추출 (엘리먼트 대기 포함)
                element = handler.wait_for_captcha()
                if element:
                    img_bytes = handler.extract_captcha_image(element)
                    
                    if img_bytes:
                        filename = self.save_image(img_bytes)
                        collected += 1
                        print(f"✅ 수집됨 ({collected}/{count}): {filename}")
                        
                        # 새로고침하여 다음 이미지 호출
                        handler.refresh_captcha()
                        time.sleep(0.5)
                    else:
                        print("⚠️ 이미지 데이터를 추출할 수 없습니다")
                        handler.refresh_captcha()
                else:
                    print("⚠️ CAPTCHA 이미지를 찾을 수 없습니다. 페이지를 새로고침합니다.")
                    self.driver.refresh()
                    time.sleep(3)
                    
        except KeyboardInterrupt:
            print("🛑 사용자에 의해 중단되었습니다")
        except Exception as e:
            print(f"❌ 수집 중 오류 발생: {e}")
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
            
        print(f"🎉 수집 완료: 총 {collected}개의 이미지")
        return collected


def main():
    """데이터 수집기 실행 진입점."""
    # .env 로드
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    crawler = CaptchaCrawler()
    # 예시: 50개 이미지 수집
    # crawler.collect(count=50)


if __name__ == "__main__":
    main()
