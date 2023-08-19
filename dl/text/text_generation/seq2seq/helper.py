
import json

def save_mapping(tokenizer_obj, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_obj.to_json(), ensure_ascii=False))

def save_config(config_json, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(config_json, ensure_ascii=False))