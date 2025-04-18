#model.py

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

def load_model_and_tokenizer(config):
    print("Loading tokenizer and model...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model_name)
    model = GPT2LMHeadModel.from_pretrained(config.model_name)

    # 패딩 토큰 설정
    special_tokens_dict = {'pad_token': '[PAD]'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens.")
    model.resize_token_embeddings(len(tokenizer))

    # 패딩 토큰 ID 설정
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
