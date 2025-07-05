import os
from PIL import Image
import requests
from urllib.parse import urlparse
from uuid import uuid4


def read_image_dims(path: str) -> tuple[int, int]:
    """
    Reads the dimensions of an image from the specified file path.

    Args:
        path (str): The file path to the image.

    Returns:
        tuple[int, int]: A tuple containing the width and height of the image.
    """

    img = Image.open(path)
    width, height = img.size
    return width, height


def list_all_images(path: str, sub_dir: bool = True):
    """
    Yields a list of all image files in the specified path.

    Args:
        path (str): The path to the folder containing the images.
        sub_dir (bool): If True, this function will recursively search all subdirectories
                        for image files. Defaults to True.

    Yields:
        str: A file path to an image file.
    """
    if sub_dir:
        for root, dirs, files in os.walk(path):
            for file in files:
                if (
                    file.endswith(".jpg")
                    or file.endswith(".jpeg")
                    or file.endswith(".png")
                ):
                    yield os.path.join(root, file)
    else:
        for file in os.listdir(path):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                yield os.path.join(path, file)


def calculate_cost(input_tokens: int, output_tokens: int, config: dict) -> float:
    """
    Calculates the cost of generating text from an image or directory of images.

    The cost is calculated as the sum of the input and output costs, which are
    calculated as the number of tokens divided by 10^6 and multiplied by the
    respective cost per token.

    Args:
        input_tokens (int): The number of input tokens.
        output_tokens (int): The number of output tokens.
        config (dict): A dictionary containing the cost per input token and
                       output token.

    Returns:
        float: The cost of generating text from the image or directory of images.
    """
    input_cost = (input_tokens / 10**6) * config["input_tokens"]
    output_cost = (output_tokens / 10**6) * config["output_tokens"]
    return input_cost + output_cost

def is_url(path: str) -> bool:
    """Check if a given path is a valid URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")



def save_image_from_url(url: str, save_dir: str = "Downloaded") -> str:
    """
    Download an image from a URL and save it locally.

    Args:
        url (str): URL of the image.
        save_dir (str): Directory to save the image.

    Returns:
        str: Path to the saved image file.
    """
    save_dir = os.path.abspath(save_dir)  # Convert to absolute path
    os.makedirs(save_dir, exist_ok=True)
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1] or ".jpg"  # Default to .jpg
    filename = f"{uuid4().hex}{ext}"
    path = os.path.join(save_dir, filename)

    response = requests.get(url)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    return path