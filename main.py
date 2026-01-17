"""
인터파크 티켓 예매 자동화 메인 스크립트.

이 스크립트는 다음 프로세스를 조율합니다:
1. 브라우저 설정 및 로그인
2. 티켓 예매 페이지 이동
3. CAPTCHA 감지 및 해결 (OCR 모델 사용)
4. 예매 절차 진행 (좌석 선택 등)
"""

import os
import time
from dotenv import load_dotenv

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 sys.path에 추가 (외부 실행 시 패키지 인식 보장)
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from automation.captcha_handler import CaptchaHandler
from captcha_ocr.predictor import CaptchaPredictor

try:
    import undetected_chromedriver as uc
except ImportError:
    print("❌ undetected-chromedriver가 설치되지 않았습니다. (pip install undetected-chromedriver)")
    exit(1)

def main():
    # .env 파일에서 환경변수 로드
    load_dotenv()
    
    # 1. 브라우저 초기화
    options = uc.ChromeOptions()
    # options.add_argument('--headless') # 필요시 헤드리스 모드 활성화
    
    driver = uc.Chrome(options=options)
    driver.implicitly_wait(10)
    
    try:
        # 2. 로그인 (환경변수 사용)
        user_id = os.getenv("INTERPARK_ID")
        user_pw = os.getenv("INTERPARK_PW")
        target_url = os.getenv("TARGET_URL")
        
        if not all([user_id, user_pw, target_url]):
            print("❌ .env 파일에 INTERPARK_ID, INTERPARK_PW, TARGET_URL을 설정해주세요.")
            return

        print("🚀 자동화 프로세스를 시작합니다...")
        
        # 로그인 페이지 이동
        driver.get("https://accounts.interpark.com/login")
        driver.find_element("id", "userId").send_keys(user_id)
        driver.find_element("id", "userPw").send_keys(user_pw)
        driver.find_element("id", "btn_login").click()
        time.sleep(2)
        
        # 3. 대상 티켓 페이지로 이동
        print(f"📍 티켓 페이지 이동 중: {target_url}")
        driver.get(target_url)
        
        # [여기서 예매 버튼 클릭 등 페이지별 커스텀 로직이 필요할 수 있습니다]
        
        # 4. CAPTCHA 해결 프로세스
        print("🔍 CAPTCHA 감지 대기 중...")
        handler = CaptchaHandler(driver)
        
        if handler.solve_captcha(max_attempts=5):
            print("✅ CAPTCHA가 성공적으로 해결되었습니다!")
            # 5. 이후 예매 절차 진행 (좌석 선택 등)
            print("🎟️ 다음 예매 단계로 진행합니다...")
        else:
            print("❌ CAPTCHA 해결에 실패했습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        # 테스트를 위해 브라우저를 바로 닫지 않음
        input("계속하려면 엔터를 누르세요 (브라우저가 종료됩니다)...")
        driver.quit()

if __name__ == "__main__":
    main()
