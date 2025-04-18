#main.py

import argparse
from config import Config
from preprocess import prepare_data
from train_model import train_model
from generate_text import generate
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str, 
                    choices=['prepare', 'train', 'generate', 'all'],
                    default='all', 
                    help='실행할 단계 선택')
args = parser.parse_args()


def main():
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M")
    Config.current_time = current_time  # Config 클래스에 시간 저장
    
    Config.create_directories()
    
    # 현재 실험 정보 출력
    print(f"\n=== 실험 시작 ===")
    print(f"실험 디렉토리: {Config.current_exp_dir}")
    print(f"처리된 데이터 저장 위치: {Config.processed_data_path}")
    print(f"모델 저장 위치: {Config.models_dir}")
    

    if args.stage in ['prepare', 'all']:
        print("\n=== 데이터 준비 단계 시작 ===")
        prepare_data()
    
    if args.stage in ['train', 'all']:
        print("\n=== 학습 단계 시작 ===")
        train_model()
    
    if args.stage in ['generate', 'all']:
        print("\n=== 텍스트 생성 단계 시작 ===")
        generate()

    print(f"\n=== 실험 완료 ===")
    print(f"결과는 {Config.current_exp_dir} 에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main()
