import os
from PIL import Image
from io import BytesIO
import requests
from requests.exceptions import HTTPError , RequestException
from image_token.caching_utils import ImageDimensionCache
from image_token.core import calculate_image_tokens


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


def get_image_dimensions_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    """
    Reads the image bytes from request

    Args:
        input_bytes(int): The number of input tokens.

    Returns:
        tuple: [height , width].
    """
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            return img.size
    except Exception as e:
        return None


def process_image_from_url(url: str, model_config: dict, cache : ImageDimensionCache , model_name: str = None ) -> int:
    """
    Process an image from a URL and calculate number of tokens, using persistent dimension cache.

    """
    try :
        dimensions = cache.get_cached_dimensions(url)

        if dimensions:
            width, height = dimensions
        else:
            response = requests.get(url)
            response.raise_for_status()

            image_bytes = response.content
            width, height = get_image_dimensions_from_bytes(image_bytes)
            cache.cache_dimensions(url, width, height)

        num_tokens = calculate_image_tokens(
            model_name=model_name,
            width=width,
            height=height,
            max_tokens=model_config["max_tokens"],
            model_config=model_config,
        )
        return num_tokens
    except HTTPError as http_err:
        print(f"HTTP error occurred while fetching image: {http_err}")
    except RequestException as req_err:
        print(f"Network error occurred: {req_err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")

    return -1


def check_case(dataset: str, config: str = "default"):
    url = f'https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset=0&length=1'
    print(f"Requesting URL: {url}")

    try:
        response = requests.get(url)
        data = response.json()

        # Api error case - dataset has the image in pil format so cause of that size is large
        if "error" in data:
            print("Case 2 - Dataset too large or server error")
            print(f"Error: {data['error']}")
            return "Case 2"

        # Api gives the json with height and widht
        features = data.get("features", [])
        image_features = [f for f in features if f.get("type", {}).get("_type") == "Image"]
        if image_features:
            print("Case 1 - Image defined in metadata")
            return "Case 1"

        # Checking if any value in the row contains an image URL
        rows = data.get("rows", [])
        if rows:
            row = rows[0].get("row", {})
            for key, value in row.items():
                if isinstance(value, str) and value.startswith("http") and value.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                    print(f"Case 3  - Found image URL in field: {key}")
                    return "Case 3"
                # Check for nested dict with URL field
                if isinstance(value, dict):
                    for v in value.values():
                        if isinstance(v, str) and v.startswith("http") and v.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                            print(f"Case 3 - Found nested image URL in field: {key}")
                            return "Case 3"

        print("Unknown case - No image metadata or image URL found")
        return "Unknown"

    except Exception as e:
        print("Case 3 - Request failed or unexpected response")
        print(f"Exception: {e}")
        return "Case 3"

def is_image_url(url):
    return isinstance(url, str) and url.startswith("http") and url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))

def get_image_url_field(row):
    for key, value in row.items():
        if is_image_url(value):
            return key
        if isinstance(value, dict):
            for v in value.values():
                if is_image_url(v):
                    return key
    return None