from image_token.validate import (
    check_if_file_or_folder_exists,
    check_if_path_is_file,
    check_allowed_extensions,
    check_valid_model,
)
from image_token.utils import read_image_dims
from image_token.core import calculate_image_tokens
from image_token.config import openai_config


def process_image(path: str, model_config: dict):
    check_allowed_extensions(path=path)

    width, height = read_image_dims(path=path)

    num_tokens = calculate_image_tokens(width=width, height=height)

    return int(num_tokens * model_config["factor"])


def get_token(model_name: str, path: str):
    check_if_file_or_folder_exists(path=path)

    if check_if_path_is_file(path=path) and check_valid_model(model_name=model_name):
        model_config = openai_config[model_name]

        num_tokens = process_image(path=path, model_config=model_config)
        print("Total number of tokens: ", num_tokens)

        return num_tokens
