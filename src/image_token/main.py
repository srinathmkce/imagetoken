from image_token.utils import calculate_cost
from image_token.config import openai_config
import asyncio
from image_token.get_tokens import GetTokens
from pathlib import Path


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


def get_token(
        model_name: str,
        path: str,
        prefix_tokens: int = 9,
        save_to: str = None,
        method: str = "sync"
        ):
    """Calculate the number of tokens in an image or directory of images.

    This function processes each image, calculates the number of tokens, and returns the total count.
    It can operate in both synchronous and asynchronous modes based on the `method` parameter.
    It also saves the results to a JSON file if `save_to` is specified.

    Args:
        model_name (str): The name of the model to use for token calculation.
        path (str): The path to the image file or directory containing images.
        prefix_tokens (int, optional): The number of tokens to add as a prefix. Defaults to 9.
        save_to (str, optional): The path to a file where the results should be saved. If None,
                                 the results are not saved to a file.
        method (str, optional): The method to use for processing images. Can be "sync" or "async".

    Returns:
        int: The total number of tokens calculated.

    Notes:
        - If the path is a file, it processes that file.
        - If the path is a directory, it processes all images in that directory.
        - The function raises a ValueError if the method is not "sync" or "async".
        - If `method` is "sync", it processes images synchronously.
        - If `method` is "async", it processes images asynchronously.
    """

    obj = GetTokens(model_name, path, prefix_tokens, save_to)
    if method == "sync":
        return obj.get_token_sync()
    elif method == "async":
        return asyncio.run(obj.get_token_async())
    else:
        raise ValueError("method must be 'sync' or 'async'")
