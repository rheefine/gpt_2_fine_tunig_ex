#prepare_data.py

import torch
from config import Config
from model import load_model_and_tokenizer
import json
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
import html
from collections import defaultdict

class TextQualityStats:
    def __init__(self):
        self.total_articles = 0
        self.total_sentences = 0
        self.processed_sentences = 0
        self.skipped_sentences = defaultdict(int)
        self.length_distribution = defaultdict(int)
    
    def print_stats(self):
        print("\n=== 텍스트 품질 통계 ===")
        print(f"전체 기사 수: {self.total_articles}")
        print(f"전체 문장 수: {self.total_sentences}")
        print(f"처리된 문장 수: {self.processed_sentences}")
        print("\n건너뛴 문장 통계:")
        for reason, count in self.skipped_sentences.items():
            print(f"- {reason}: {count}")
        print("\n문장 길이 분포:")
        for length, count in sorted(self.length_distribution.items()):
            print(f"- {length*50}-{(length+1)*50} 글자: {count}개")


def clean_text(text):
    # HTML 태그 제거
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # HTML 특수 문자 디코딩
    text = html.unescape(text)
    
    # 특수문자 처리
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # HTML 엔터티 제거
    text = re.sub(r'[\u201c\u201d]', '"', text)  # 스마트 따옴표 변환
    text = re.sub(r'[\u2018\u2019]', "'", text)  # 스마트 작은따옴표 변환
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 불필요한 특수 문자 제거 (숫자 관련 특수문자 포함)
    text = re.sub(r'[^가-힣a-zA-Z0-9.,!?%()"\'\s-]', '', text)
    
    return text

def protect_special_patterns(text: str) -> str:
    """특수 패턴들(소수점, 약어 등)을 임시 토큰으로 치환"""
    patterns = [
        # (정규표현식 패턴, 치환할 마커, lambda 사용 여부)
        (r'([\d,]+)\.(\d+)', '@DECIMAL@', False),             # 소수점 숫자
        (r'\b(주식회사|\(주\)|㈜)', '@ABV@', True),            # 기업명 약어
        (r'[\w\.-]+@[\w\.-]+', '@EMAIL@', True),              # 이메일
        (r'"[^"]*"', '@QUOTE@', True),                        # 인용문
        (r'\'[^\']*\'', '@SQUOTE@', True),                    # 작은따옴표
        (r'\([^\)]*\)', '@PAREN@', True)                      # 괄호
    ]
    
    protected_text = text
    for pattern, marker, use_lambda in patterns:
        if use_lambda:
            protected_text = re.sub(pattern, 
                lambda m: m.group().replace('.', marker), 
                protected_text)
        else:
            protected_text = re.sub(pattern, rf'\1{marker}\2', protected_text)
    
    return protected_text

def check_sentence_quality(sentence, stats):
    """문장 품질 검사"""
    if len(sentence.strip()) < 10:
        stats.skipped_sentences['너무 짧은 문장'] += 1
        return False
        
    if len(sentence.strip()) > 200:
        stats.skipped_sentences['너무 긴 문장'] += 1
        return False
    
    if not any(ch.isalpha() for ch in sentence):
        stats.skipped_sentences['텍스트 없음'] += 1
        return False
    
    if sentence.count('"') % 2 != 0 or sentence.count('"') % 2 != 0:
        stats.skipped_sentences['따옴표 쌍 불일치'] += 1
        return False
    
    # 의미없는 반복 패턴 체크
    if any(char * 3 in sentence for char in '.。,，!！?？'):
        stats.skipped_sentences['과도한 문장부호'] += 1
        return False
        
    # 명확한 불완전 문장만 필터링
    if re.search(r'\d+\.$', sentence.strip()):  # 숫자로 끝나는 경우만
        stats.skipped_sentences['불완전한 문장 끝'] += 1
        return False

    # 문장 길이 분포 기록
    length_bucket = len(sentence) // 50
    stats.length_distribution[length_bucket] += 1
    
    stats.processed_sentences += 1
    return True

def prepare_data():    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(Config)
    
    print(f"데이터 로드 중: {Config.data_path}")
    with open(Config.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = TextQualityStats()
    stats.total_articles = len(data)
    
    samples = []
    
    for item in tqdm(data, desc="Processing articles"):
        # 텍스트 전처리
        text = clean_text(item['body'])

        protected_text = protect_special_patterns(text)

        # 문장 단위로 분리하여 처리
        sentences = re.split(r'([.!?]+)\s+', protected_text)
        stats.total_sentences += len(sentences) // 2
        
        # 품질 검사를 통과한 문장만 모으기
        quality_sentences = []

        markers = {
            '@DECIMAL@': '.',
            '@ABV@': '.',
            '@EMAIL@': '.',
            '@QUOTE@': '.',
            '@SQUOTE@': '.',
            '@PAREN@': '.'
        }

        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
            for marker, original in markers.items():
                sentence = sentence.replace(marker, original)

            if check_sentence_quality(sentence, stats):
                quality_sentences.append(sentence)
        
        # 품질 검사를 통과한 문장들 합치기
        if quality_sentences:
            text = ' '.join(quality_sentences)
            encoded = tokenizer.encode_plus(
                text,
                max_length=Config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            samples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            })
    
    # 통계 출력
    stats.print_stats()
    
    # 처리된 데이터 저장
    torch.save(samples, Config.processed_data_path)
    print(f"처리된 데이터가 저장되었습니다: {Config.processed_data_path}")
    print(f"최종 샘플 수: {len(samples)}")
    
    return samples, tokenizer

if __name__ == "__main__":
    prepare_data() 
