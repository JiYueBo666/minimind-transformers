from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('tokenizer/')
print('bos', t.bos_token, t.bos_token_id)
print('eos', t.eos_token, t.eos_token_id)
print('pad', t.pad_token, t.pad_token_id)