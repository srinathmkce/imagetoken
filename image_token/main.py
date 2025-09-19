from image_token.openai import OpenAiModel
from image_token.gemini import GeminiModel
from image_token.config import openai_config , gemini_config
from pathlib import Path

_current_model = None

def set_model(name: str):
    global _current_model
    _current_model = name.lower()

def _get_model(model_name):
    if model_name in openai_config:
        return OpenAiModel()
    elif model_name in gemini_config:
        return GeminiModel()
    else:
        raise ValueError(f"Unsupported model: {_current_model}")

def get_token(model_name: str, path: str|Path, save_to: str = None , **kwargs):
    model = _get_model(model_name)
    return model.get_token(model_name=model_name , path = path , save_to=save_to , **kwargs)

def get_cost(model_name: str,system_prompt_tokens: int,approx_output_tokens: int,path: Path | str,save_to: str = None, **kwargs):
    model = _get_model(model_name)
    return model.get_cost(
        model_name,
        system_prompt_tokens,
        approx_output_tokens,
        path,
        save_to,
        **kwargs
    )
