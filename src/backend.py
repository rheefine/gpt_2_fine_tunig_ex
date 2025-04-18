from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from pathlib import Path

app = Flask(__name__)
CORS(app)

class ModelConfig:
    model_name = "skt/kogpt2-base-v2"
    model_path = Path(f"../model/2024_11_24_1611/models/best_model.pt")

class TextPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        print("Loading tokenizer and model...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(ModelConfig.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(ModelConfig.model_name)

        special_tokens_dict = {'pad_token': '[PAD]'}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens.")
        self.model.resize_token_embeddings(len(self.tokenizer))

        if ModelConfig.model_path.exists():
            self.model.load_state_dict(torch.load(ModelConfig.model_path))
            print(f"Loaded trained model from {ModelConfig.model_path}")
        else:
            print(f"Warning: No trained model found at {ModelConfig.model_path}")
            print("Using base model without fine-tuning")

        self.model.to(self.device)
        self.model.eval()

    def predict_next_word(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            
            # 특수 토큰 필터링
            for id in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, 
                      self.tokenizer.bos_token_id, self.tokenizer.unk_token_id]:
                if id is not None:
                    next_token_logits[id] = float('-inf')
            
            # 특수 문자 필터링
            for id in range(len(self.tokenizer)):
                token = self.tokenizer.decode([id])
                if len(token.strip()) <= 1 and not token.isalnum():
                    next_token_logits[id] = float('-inf')
            
            # Top-k 후보 선택
            top_k = 5
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # 후보 중 가장 높은 확률의 단어 선택
            for idx in top_k_indices:
                word = self.tokenizer.decode([idx]).strip()
                if (len(word) > 1 or word.isalnum()) and not any(c in word for c in "[]<>"):
                    return word
            
            return "[예측 실패]"

# 전역 predictor 인스턴스 생성
predictor = TextPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': '텍스트가 필요합니다'}), 400
            
        # 다음 단어 예측
        prediction = predictor.predict_next_word(text)
        
        return jsonify({
            'prediction': prediction,
            'input_text': text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
