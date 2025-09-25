import os
from image_token.utils.config import openai_config
from urllib.parse import urlparse


def check_if_file_or_folder_exists(path: str):
    """
    Checks if a given path exists as a file or directory.

    Args:
        path (str): The path to check.

    Raises:
        FileNotFoundError: If the path does not exist.

    Returns:
        bool: True if the path exists, False otherwise.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    else:
        return True


def check_if_path_is_file(path: str):
    """
    Checks if a given path is a file.

    Args:
        path (str): The path to check

    Returns:
        bool: True if the path is a file, False otherwise

    """
    try:
        return os.path.isfile(path)
    except:
        return False
    # return os.path.isfile(path)


def check_if_path_is_folder(path: str) -> bool:
    """
    Checks if the given path is a folder (directory).

    Args:
        path (str): The path to check

    Returns:
        bool: True if the path is a folder, False otherwise
    """
    try:
        return os.path.isdir(path)
    except:
        return False


def is_url(path: str) -> bool:
    """Check if a given path is a valid URL."""
    # parsed = urlparse(path)
    # return parsed.scheme in ("http", "https")
    try:
        parsed = urlparse(path)
        return parsed.scheme in ("http", "https")
    except:
        return False


def check_allowed_extensions(path: str):
    """
    Checks if the file at the given path has an allowed extension.

    Args:
        path (str): The path to the file.

    Raises:
        ValueError: If the file does not have a valid extension.
    """

    allowed_extensions = [".jpg", ".jpeg", ".png"]
    if not any(path.endswith(ext) for ext in allowed_extensions):
        raise ValueError(f"Invalid file extension: {path}")


def check_valid_model(model_name: str):
    """
    Checks if the given model name is valid.

    Args:
        model_name (str): The model name to check.

    Raises:
        ValueError: If the model name is not valid.

    Returns:
        bool: True if the model name is valid, False otherwise.

    """
    if model_name not in openai_config.keys():
        raise ValueError(
            f"Invalid model name: {model_name}. Only supported models are : {list(openai_config.keys())}"
        )
    else:
        return True


def is_multiple_urls(urls) -> bool:
    """

    Checks if multiple urls are passed

    """
    try:
        if len(urls) > 0 and is_url(urls[0]):
            return True
        else:
            return False
    except:
        return False
