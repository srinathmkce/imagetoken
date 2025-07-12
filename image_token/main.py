import json
from image_token.validate import (
    check_if_path_is_file,
    check_if_path_is_folder,
    check_allowed_extensions,
    check_valid_model,
    is_url,
    is_multiple_urls
)
from image_token.utils import (
    read_image_dims,
    calculate_cost,
    list_all_images,
    process_image_from_url,
)
from image_token.core import calculate_image_tokens
from image_token.config import openai_config
from tqdm import tqdm
from pathlib import Path
from image_token.caching_utils import ImageDimensionCache


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


def get_token(model_name: str, path: str, prefix_tokens: int = 9 , save_to: str = None ):
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

    if check_if_path_is_file(path=path):
        num_tokens = process_image(
            path=path, model_config=model_config, model_name=model_name
        )
        total_tokens = num_tokens + prefix_tokens
        result_dict[path] = total_tokens
    elif check_if_path_is_folder(path=path):
        image_files = list_all_images(path=path)
        total_tokens = 0
        for image_path in tqdm(image_files):
            num_tokens = process_image(
                path=image_path, model_config=model_config, model_name=model_name
            )
            total_tokens += num_tokens + prefix_tokens
            result_dict[image_path] = num_tokens + prefix_tokens
    elif is_url(path=path):
        with ImageDimensionCache() as cache:
            num_tokens = process_image_from_url(
            url=path,
            model_config=model_config,
            model_name=model_name,
            cache = cache
        )
        total_tokens = num_tokens + prefix_tokens
        result_dict[path] = total_tokens
    elif is_multiple_urls(urls = path):
        with ImageDimensionCache() as cache:
            for url in path : 
                num_tokens = process_image_from_url(
                    url = url,
                    model_name=model_name,
                    model_config=model_config,
                    cache=cache
                )
                print(num_tokens)
                total_tokens += num_tokens + prefix_tokens
                result_dict[url] = num_tokens + prefix_tokens
    else:
        raise ValueError(
            f"Invalid input path or URL: '{path}'. The given input is not a valid file, folder, or URL."
        )

    if save_to:
        with open(save_to, "w") as f:
            json.dump(result_dict, f, indent=4)

    return total_tokens
