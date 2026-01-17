"""
스마트 CAPTCHA 어노테이션 라벨링 GUI.

주요 기능:
    - 이미지 표시 및 확대
    - EasyOCR 예측 결과 참고용 표시
    - 커스텀 모델 예측 결과 표시
    - 입력값 유효성 검사 (대문자 6자리)
    - 일정 수량 누적 시 자동 학습 트리거
    - 효율적인 작업을 위한 키보드 단축키
"""

import os
import re
import subprocess
from typing import Optional, Callable
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    raise ImportError("Tkinter is required. Install with: brew install python-tk")

try:
    import easyocr
except ImportError:
    easyocr = None

from .label_manager import LabelManager


class LabelingApp:
    """
    효율적인 CAPTCHA 라벨링을 위한 GUI 애플리케이션 클래스.
    
    애플리케이션은 CAPTCHA 이미지를 표시하고, 사용자가 OCR 예측 결과를
    참고하여 라벨을 입력할 수 있도록 돕습니다. 라벨링된 샘플 수를 추적하며,
    설정된 임계값 도달 시 자동으로 모델 학습을 시작할 수 있습니다.
    
    사용 예시:
        >>> app = LabelingApp()
        >>> app.run()
    """
    
    TRAINING_THRESHOLD = 50  # 50개 이상의 새로운 라벨이 쌓이면 학습 트리거
    
    def __init__(
        self,
        image_dir: str = "data/raw",
        label_manager: LabelManager = None,
        model_predictor: Callable = None
    ):
        """
        라벨링 애플리케이션을 초기화합니다.
        
        Args:
            image_dir: 라벨링되지 않은 이미지가 있는 디렉토리
            label_manager: 라벨 관리 인스턴스
            model_predictor: 커스텀 모델 예측을 위한 선택적 함수
        """
        self.image_dir = image_dir
        self.label_manager = label_manager or LabelManager()
        self.model_predictor = model_predictor
        
        # 필수 디렉토리 확인 및 생성
        os.makedirs(self.image_dir, exist_ok=True)
        
        # EasyOCR 리더 초기화
        self.ocr_reader = None
        if easyocr:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception as e:
                print(f"⚠️ EasyOCR 초기화 실패: {e}")
        
        # 이미지 목록 로드
        self.image_files = self._get_unlabeled_images()
        self.current_index = 0
        
        # GUI 설정
        self.root = None
        self._setup_gui()
    
    def _get_unlabeled_images(self) -> list:
        """라벨링되지 않은 이미지 목록을 가져옵니다 (숫자로 된 파일명)."""
        if not os.path.exists(self.image_dir):
            return []
        
        files = []
        for f in sorted(os.listdir(self.image_dir)):
            if f.endswith('.png'):
                # 숫자로 된 파일명만 라벨링 대상으로 간주 (미라벨링 상태)
                name = f[:-4]
                if name.isdigit():
                    files.append(f)
        
        return files
    
    def _setup_gui(self) -> None:
        """GUI 컴포넌트들을 설정합니다."""
        self.root = tk.Tk()
        self.root.title("🏷️ Smart CAPTCHA Labeling Tool")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Progress label
        self.progress_label = tk.Label(
            main_frame,
            text="",
            font=("Helvetica", 10),
            fg="gray"
        )
        self.progress_label.pack()
        
        # Image display
        self.image_label = tk.Label(main_frame, bg='white', relief='solid', bd=1)
        self.image_label.pack(pady=10)
        
        # OCR predictions frame
        pred_frame = tk.Frame(main_frame)
        pred_frame.pack(pady=5)
        
        self.easyocr_label = tk.Label(
            pred_frame,
            text="",
            font=("Helvetica", 12),
            fg="green"
        )
        self.easyocr_label.pack()
        
        self.model_label = tk.Label(
            pred_frame,
            text="",
            font=("Helvetica", 12),
            fg="blue"
        )
        self.model_label.pack()
        
        # Input field
        self.input_var = tk.StringVar()
        self.input_var.trace_add("write", self._on_input_change)
        
        self.entry = tk.Entry(
            main_frame,
            textvariable=self.input_var,
            font=("Helvetica", 18),
            justify='center',
            width=10
        )
        self.entry.pack(pady=10)
        self.entry.bind('<KeyRelease>', self._force_uppercase)
        
        # Warning label
        self.warning_label = tk.Label(
            main_frame,
            text="",
            font=("Helvetica", 10),
            fg="red"
        )
        self.warning_label.pack()
        
        # Buttons frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        self.skip_btn = tk.Button(
            btn_frame,
            text="⏭️ Skip",
            command=self._skip_image,
            width=10
        )
        self.skip_btn.pack(side='left', padx=5)
        
        self.next_btn = tk.Button(
            btn_frame,
            text="✅ Next (Enter)",
            command=self._save_and_next,
            width=15,
            state='disabled'
        )
        self.next_btn.pack(side='left', padx=5)
        
        # Statistics label
        self.stats_label = tk.Label(
            main_frame,
            text="",
            font=("Helvetica", 9),
            fg="gray"
        )
        self.stats_label.pack(pady=5)
        
        # Keyboard bindings
        self.root.bind('<Return>', lambda e: self._save_and_next() if self.next_btn['state'] == 'normal' else None)
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
        # Load first image
        self._load_current_image()
        self._update_stats()
    
    def _force_uppercase(self, event=None) -> None:
        """입력값을 대문자로 강제 변환합니다."""
        value = self.input_var.get()
        # 영문자가 아닌 문자는 제거하고 대문자로 변환
        cleaned = ''.join(c for c in value.upper() if c.isalpha())
        if cleaned != value:
            self.input_var.set(cleaned)
    
    def _on_input_change(self, *args) -> None:
        """입력값 변경 시 유효성 검사를 수행합니다."""
        value = self.input_var.get().strip().upper()
        
        if re.fullmatch(r'[A-Z]{6}', value):
            self.next_btn.config(state='normal')
            self.warning_label.config(text="")
        else:
            self.next_btn.config(state='disabled')
            if value:
                remaining = 6 - len(value)
                if remaining > 0:
                    self.warning_label.config(text=f"⚠️ {remaining}글자가 더 필요합니다")
                else:
                    self.warning_label.config(text="⚠️ 영문 대문자만 입력 가능합니다")
            else:
                self.warning_label.config(text="")
    
    def _load_current_image(self) -> None:
        """현재 이미지를 로드하여 화면에 표시합니다."""
        if self.current_index >= len(self.image_files):
            self._show_completion()
            return
        
        filename = self.image_files[self.current_index]
        img_path = os.path.join(self.image_dir, filename)
        
        # 진행 상태 업데이트
        self.progress_label.config(
            text=f"이미지 {self.current_index + 1} / {len(self.image_files)}"
        )
        
        # 이미지 로드
        try:
            image = Image.open(img_path)
            image = image.resize((300, 100), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {e}")
            self._skip_image()
            return
        
        # EasyOCR 예측 결과 가져오기
        if self.ocr_reader:
            try:
                results = self.ocr_reader.readtext(img_path, detail=0, paragraph=False)
                prediction = ''.join(results).strip().upper()
                prediction = ''.join(c for c in prediction if c.isalpha())[:6]
                self.easyocr_label.config(text=f"🟢 EasyOCR 예측: {prediction or '(없음)'}")
            except Exception as e:
                self.easyocr_label.config(text=f"🟢 EasyOCR 예측: (에러)")
        else:
            self.easyocr_label.config(text="🟢 EasyOCR 예측: (사용 불가)")
        
        # 커스텀 모델 예측 결과 가져오기
        if self.model_predictor:
            try:
                model_pred = self.model_predictor(img_path)
                self.model_label.config(text=f"🔵 모델 예측: {model_pred}")
            except Exception as e:
                self.model_label.config(text=f"🔵 모델 예측: (에러)")
        else:
            self.model_label.config(text="🔵 모델 예측: (모델 미로드)")
        
        # 입력창 초기화 및 포커스
        self.input_var.set("")
        self.entry.focus()
    
    def _save_and_next(self) -> None:
        """라벨을 저장하고 다음 이미지로 이동합니다."""
        label = self.input_var.get().strip().upper()
        filename = self.image_files[self.current_index]
        
        try:
            new_filename = self.label_manager.add_label(filename, label)
            print(f"✅ {filename} → {new_filename} ({label})")
        except Exception as e:
            messagebox.showerror("에러", f"라벨 저장 실패: {e}")
            return
        
        # 학습 트리거 확인
        untrained = self.label_manager.count_untrained()
        if untrained >= self.TRAINING_THRESHOLD:
            self._trigger_training()
        
        # 다음 단계로 이동
        self.current_index += 1
        self._load_current_image()
        self._update_stats()
    
    def _skip_image(self) -> None:
        """현재 이미지를 라벨링하지 않고 건너뜁니다."""
        self.current_index += 1
        self._load_current_image()
    
    def _update_stats(self) -> None:
        """통계 정보 표시를 업데이트합니다."""
        stats = self.label_manager.get_statistics()
        self.stats_label.config(
            text=f"전체: {stats['total']} | 학습됨: {stats['trained']} | 대기중: {stats['untrained']}"
        )
    
    def _trigger_training(self) -> None:
        """모델 학습을 백그라운드에서 트리거합니다."""
        print(f"🚀 학습 임계값 도달 ({self.TRAINING_THRESHOLD}개)")
        print("🔄 백그라운드 학습 시작...")
        
        # 학습 프로세스 시작
        try:
            subprocess.Popen(
                ["python3", "-m", "captcha_ocr.trainer"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"⚠️ 학습 시작 실패: {e}")
    
    def _show_completion(self) -> None:
        """모든 이미지 라벨링 완료 시 메시지를 표시합니다."""
        self.image_label.config(image='')
        self.easyocr_label.config(text="")
        self.model_label.config(text="")
        self.warning_label.config(text="")
        self.entry.config(state='disabled')
        self.next_btn.config(state='disabled')
        self.skip_btn.config(state='disabled')
        
        self.progress_label.config(
            text="🎉 모든 이미지 라벨링 완료!",
            font=("Helvetica", 14),
            fg="green"
        )
        
        messagebox.showinfo(
            "완료",
            "모든 이미지의 라벨링이 끝났습니다!\n\n" + 
            f"총 라벨링 개수: {self.label_manager.get_statistics()['total']}"
        )
    
    def run(self) -> None:
        """라벨링 애플리케이션을 실행합니다."""
        if not self.image_files:
            print("⚠️ 라벨링할 새 이미지가 없습니다:", self.image_dir)
            return
        
        print(f"📂 {len(self.image_files)}개의 미라벨링 이미지 발견")
        print("🏷️ 라벨링 툴 시작 중...")
        
        self.root.mainloop()


def main():
    """라벨링 툴 실행 진입점."""
    app = LabelingApp(
        image_dir="data/raw"
    )
    app.run()


if __name__ == "__main__":
    main()
