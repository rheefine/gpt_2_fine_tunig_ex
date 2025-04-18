#config.py

from pathlib import Path

class Config:
   # 실험 시간 (main에서 설정)
   current_time = None

   # 기본 경로 설정
   base_path = Path(__file__).parent.parent
   data_path = base_path / "data/data_ex.json"
   
   # 결과 디렉토리 내부 경로
   result_dir = base_path / "model"
   
   # 데이터 설정
   max_length = 256        # 텍스트 최대 길이
   batch_size = 16         # 배치 크기 (GPU 메모리에 맞게 조정)
   num_train_epochs = 5    # 학습 에포크
   learning_rate = 5e-5    # 학습률
   model_name = "skt/kogpt2-base-v2"

   @classmethod
   def create_directories(cls):
       """필요한 모든 디렉토리를 순서대로 생성"""
       # 현재 실험 디렉토리 설정
       cls.current_exp_dir = cls.result_dir / cls.current_time
       cls.models_dir = cls.current_exp_dir / "models"
       cls.processed_data_path = cls.current_exp_dir / "processed_dataset.pt"
       cls.best_model_path = cls.models_dir / "best_model.pt"
       
       # 디렉토리 생성
       cls.result_dir.mkdir(parents=True, exist_ok=True)
       cls.current_exp_dir.mkdir(parents=True, exist_ok=True)
       cls.models_dir.mkdir(parents=True, exist_ok=True)
