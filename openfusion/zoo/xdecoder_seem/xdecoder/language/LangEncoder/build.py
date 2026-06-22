import os

from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from .registry import lang_encoders
from .registry import is_lang_encoder


def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']

    if not is_lang_encoder(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return lang_encoders(model_name)(config_encoder, tokenizer, verbose, **kwargs)


def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if config_encoder['TOKENIZER'] == 'clip':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        try:
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer, cache_dir='./cache')
        except OSError:
            tokenizer_path = snapshot_download(
                pretrained_tokenizer,
                cache_dir='./cache',
                allow_patterns=[
                    'vocab.json',
                    'merges.txt',
                    'tokenizer.json',
                    'tokenizer_config.json',
                    'special_tokens_map.json',
                    'config.json',
                ],
            )
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    elif config_encoder['TOKENIZER'] == 'clip-fast':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        try:
            tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_tokenizer, from_slow=True, cache_dir='./cache')
        except OSError:
            tokenizer_path = snapshot_download(
                pretrained_tokenizer,
                cache_dir='./cache',
                allow_patterns=[
                    'vocab.json',
                    'merges.txt',
                    'tokenizer.json',
                    'tokenizer_config.json',
                    'special_tokens_map.json',
                    'config.json',
                ],
            )
            tokenizer = CLIPTokenizerFast.from_pretrained(tokenizer_path, from_slow=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_encoder['TOKENIZER'], cache_dir='./cache')

    return tokenizer
