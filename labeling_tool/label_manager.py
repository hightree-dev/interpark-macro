"""CSV 기반 라벨 저장 및 관리를 위한 모듈."""

import os
import csv
import shutil
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class LabelEntry:
    """개별 라벨 항목 클래스."""
    filename: str
    label: str
    trained: bool = False


class LabelManager:
    """
    CSV 형식의 CAPTCHA 라벨을 관리하는 클래스.
    
    주요 기능:
        - 라벨 추가 및 업데이트
        - 학습 상태(trained) 추적
        - 라벨 기반 파일 자동 이름 변경
        - 미학습 샘플 개수 확인
    """
    
    CSV_HEADERS = ["filename", "label", "trained"]
    
    def __init__(
        self,
        csv_path: str = "data/labels.csv",
        image_dir: str = "data/raw",
        labeled_dir: str = "data/labeled"
    ):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.labeled_dir = labeled_dir
        
        # Create directories if needed
        os.makedirs(self.labeled_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        
        # Initialize CSV if not exists
        self._init_csv()
    
    def _init_csv(self) -> None:
        """CSV 파일이 없을 경우 헤더와 함께 초기화합니다."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)
    
    def add_label(
        self,
        original_filename: str,
        label: str,
        copy_to_labeled: bool = True
    ) -> str:
        """
        새로운 라벨 항목을 추가합니다.
        
        Args:
            original_filename: 원본 이미지 파일명
            label: CAPTCHA 텍스트 라벨 (대문자 6자리)
            copy_to_labeled: labeled 디렉토리로 복사 여부
            
        Returns:
            이름 변경 후의 새 파일명
        """
        # 라벨 유효 검사
        label = label.strip().upper()
        if not self._validate_label(label):
            raise ValueError(f"유효하지 않은 라벨입니다: {label}. 영어 대문자 6자리여야 합니다.")
        
        # 고유 파일명 생성
        new_filename = self._get_unique_filename(label)
        
        # 파일 이름 변경/이동
        old_path = os.path.join(self.image_dir, original_filename)
        new_path = os.path.join(self.image_dir, new_filename)
        
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
        
        # labeled 디렉토리로 복사
        if copy_to_labeled:
            labeled_path = os.path.join(self.labeled_dir, new_filename)
            shutil.copy2(new_path, labeled_path)
        
        # CSV에 추가
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([new_filename, label, ""])
        
        return new_filename
    
    def _validate_label(self, label: str) -> bool:
        """라벨 형식이 올바른지 확인합니다 (대문자 6자리)."""
        return len(label) == 6 and label.isalpha() and label.isupper()
    
    def _get_unique_filename(self, label: str) -> str:
        """라벨을 기반으로 중복되지 않는 파일명을 생성합니다."""
        base_name = f"{label}.png"
        path = os.path.join(self.image_dir, base_name)
        
        if not os.path.exists(path):
            return base_name
        
        # 중복 시 접미사 추가
        count = 1
        while True:
            new_name = f"{label}_{count}.png"
            path = os.path.join(self.image_dir, new_name)
            if not os.path.exists(path):
                return new_name
            count += 1
    
    def count_untrained(self) -> int:
        """학습에 아직 사용되지 않은 샘플 개수를 반환합니다."""
        count = 0
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('trained', '').lower() != 'yes':
                    count += 1
        return count
    
    def mark_as_trained(self, filenames: List[str]) -> None:
        """샘플들을 학습 완료 처리합니다."""
        # 모든 항목 읽기
        rows = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['filename'] in filenames:
                    row['trained'] = 'yes'
                rows.append(row)
        
        # 다시 쓰기
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
    
    def get_untrained_entries(self) -> List[LabelEntry]:
        """모든 미학습 라벨 항목을 반환합니다."""
        entries = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('trained', '').lower() != 'yes':
                    entries.append(LabelEntry(
                        filename=row['filename'],
                        label=row['label'],
                        trained=False
                    ))
        return entries
    
    def get_all_entries(self) -> List[LabelEntry]:
        """모든 라벨 항목을 반환합니다."""
        entries = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(LabelEntry(
                    filename=row['filename'],
                    label=row['label'],
                    trained=row.get('trained', '').lower() == 'yes'
                ))
        return entries
    
    def get_statistics(self) -> Dict[str, int]:
        """라벨링 통계를 반환합니다."""
        entries = self.get_all_entries()
        total = len(entries)
        trained = sum(1 for e in entries if e.trained)
        
        return {
            "total": total,
            "trained": trained,
            "untrained": total - trained
        }
