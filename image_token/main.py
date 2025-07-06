import json
from image_token.validate import (
    check_if_file_or_folder_exists,
    check_if_path_is_file,
    check_allowed_extensions,
    check_valid_model,
)
from image_token.utils import read_image_dims, calculate_cost, list_all_images , get_image_dimensions_from_bytes , is_url
from image_token.core import calculate_image_tokens
from image_token.config import openai_config
from tqdm import tqdm
from pathlib import Path
import requests
from image_token.caching_utils import get_cached_dimensions , cache_dimensions
# import time


def process_image(path: str, model_config: dict, model_name: str = None):
    """
    Process an image and calculate the number of tokens in it.

    Args:
        path (str): The path to the image file.
        model_config (dict): The configuration for the model.

    Returns:
        int: The number of tokens in the image.
    """
    check_allowed_extensions(path=path)

    width, height = read_image_dims(path=path)

    max_tokens = model_config["max_tokens"]

    num_tokens = calculate_image_tokens(
        model_name=model_name,
        width=width,
        height=height,
        max_tokens=max_tokens,
        model_config=model_config,
    )

    return num_tokens

def process_image_from_url(url: str, model_config: dict, model_name: str = None) -> int:
    """
    Process an image from a URL and calculate number of tokens, using persistent dimension cache.
    
    """
    # start = time.time()


    dimensions = get_cached_dimensions(url)

    if dimensions:
        print("Using cached Dimension")
        width, height = dimensions
    else:
        response = requests.get(url)
        response.raise_for_status()

        image_bytes = response.content
        width, height = get_image_dimensions_from_bytes(image_bytes)
        cache_dimensions(url, width, height)

    num_tokens = calculate_image_tokens(
        model_name=model_name,
        width=width,
        height=height,
        max_tokens=model_config["max_tokens"],
        model_config=model_config,
    )
    # end = time.time()
    # print("Token calc time:", end - start)

    return num_tokens


def get_cost(
    model_name: str,
    system_prompt_tokens: int,
    approx_output_tokens: int,
    path: Path | str,
    save_to: str = None,
    prefix_tokens: int = 9,
):
    """
    Calculate and return the estimated cost of generating text from an image or directory of images.

    Args:
        model_name (str): The name of the model to use.
        system_prompt_tokens (int): The number of tokens in the system prompt.
        approx_output_tokens (int): The approximate number of tokens in the output.
        path (str): The path to the image file or directory of images.
        save_to (str): The path to save the output to.
        prefix_tokens (int): The number of prefix tokens to use. Defaults to 9.

    Returns:
        float: The estimated cost in dollars.
    """

    model_config = openai_config[model_name]
    input_tokens = get_token(
        model_name=model_name, path=path, prefix_tokens=prefix_tokens, save_to=save_to
    )
    cost = calculate_cost(
        input_tokens=system_prompt_tokens + input_tokens,
        output_tokens=approx_output_tokens,
        config=model_config,
    )
    return cost


def get_token(model_name: str, path: str, prefix_tokens: int = 9, save_to: str = None):
    """
    Calculate and return the total number of tokens for a given image or directory of images.

    This function processes an image or a directory of images, calculates the number of tokens
    for each image based on the given model configuration, and optionally saves the results to
    a file. The total number of tokens includes the prefix tokens.

    Args:
        model_name (str): The name of the model to use for token calculation.
        path (str): The path to the image file or directory containing images.
        prefix_tokens (int, optional): The number of tokens to add as a prefix. Defaults to 9.
        save_to (str, optional): The path to a file where the results should be saved. If None,
                                 the results are not saved to a file.

    Returns:
        int: The total number of tokens calculated.
    """

    check_valid_model(model_name=model_name)
    model_config = openai_config[model_name]
    result_dict = {}
    total_tokens = 0

    if is_url(path):
        num_tokens = process_image_from_url(url=path , model_config=model_config, model_name=model_name)
        total_tokens = num_tokens + prefix_tokens
        result_dict[path] = total_tokens

    else:
        check_if_file_or_folder_exists(path=path)

        if check_if_path_is_file(path=path):
            num_tokens = process_image(
                path=path, model_config=model_config, model_name=model_name
            )
            total_tokens = num_tokens + prefix_tokens
            result_dict[path] = total_tokens

        else:
            image_files = list_all_images(path=path)
            for image_path in tqdm(image_files):
                num_tokens = process_image(
                    path=image_path, model_config=model_config, model_name=model_name
                )
                total_tokens += num_tokens + prefix_tokens
                result_dict[image_path] = num_tokens + prefix_tokens

        if save_to:
            with open(save_to, "w") as f:
                json.dump(result_dict, f, indent=4)

    return total_tokens
