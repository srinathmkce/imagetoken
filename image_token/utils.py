import os
from PIL import Image
from io import BytesIO
import requests
from requests.exceptions import HTTPError, RequestException
from image_token.caching_utils import ImageDimensionCache
import tiktoken

def calculate_text_tokens(model_name: str, text: str):
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(text)
    num_tokens = len(tokens)
    return num_tokens

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


def get_image_dimensions_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    """
    Reads the image bytes from request

    Args:
        input_bytes(int): The number of input tokens.

    Returns:
        tuple: [height , width].
    """
    try:
        print("bytes called")
        with Image.open(BytesIO(image_bytes)) as img:
            return img.size
    except Exception as e:
        return None

