#train_maodel.py

from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
from config import Config
from model import load_model_and_tokenizer

class ProcessedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_model():    
    # 데이터 로드
    print("Loading processed data...")
    samples = torch.load(Config.processed_data_path)
    dataset = ProcessedDataset(samples)
    
    # 모델과 토크나이저 로드
    model, _ = load_model_and_tokenizer(Config)
    
    # 데이터로더 생성
    train_dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 학습
    model = train(model, train_dataloader, Config)
    
    # 모델 저장
    torch.save(model.state_dict(), Config.best_model_path)
    
    return model

def train(model, dataloader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_train_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_train_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config.num_train_epochs}, Loss: {avg_loss:.4f}")

    return model
