#generate_text.py

import torch
from config import Config
from model import load_model_and_tokenizer

def generate():
    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(Config)
    model.load_state_dict(torch.load(Config.best_model_path))
    
    test_texts = [
        "대한민국의 수도는",
        "가상통화 시장은",
        "2022년은",
        "테라루나 사태는"
    ]
    
    for test_text in test_texts:
        next_word = predict_next_word(model, tokenizer, test_text)
        print(f"Input: {test_text}")
        print(f"Next word: {next_word}\n")

def predict_next_word(model, tokenizer, text):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    model.eval()
    with torch.no_grad():
        # 다음 토큰의 확률 분포 얻기
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        
        # 특수 토큰들의 확률을 매우 낮게 설정
        for id in [tokenizer.pad_token_id, tokenizer.eos_token_id, 
                  tokenizer.bos_token_id, tokenizer.unk_token_id]:
            if id is not None:
                next_token_logits[id] = float('-inf')
        
        # 특수 문자나 단일 문자에 해당하는 토큰의 확률을 낮게 설정
        for id in range(len(tokenizer)):
            token = tokenizer.decode([id])
            if len(token.strip()) <= 1 and not token.isalnum():  # 특수문자나 단일 문자
                next_token_logits[id] = float('-inf')
        
        # Top-k 후보 중에서 선택
        top_k = 100
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
        
        # 각 토큰 디코딩 및 필터링
        candidates = []
        for idx in top_k_indices:
            word = tokenizer.decode([idx]).strip()
            # 의미 있는 단어만 후보로 선정
            if (len(word) > 1 or word.isalnum()) and not any(c in word for c in "[]<>"):
                candidates.append((word, next_token_logits[idx].item()))
        
        # 후보가 없으면 다시 시도
        if not candidates:
            return "[예측 실패]"
        
        # 가장 높은 확률의 의미 있는 단어 반환
        return candidates[0][0]

if __name__ == "__main__":
    generate()
