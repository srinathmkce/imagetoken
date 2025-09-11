from image_token.main import VisionModel
import json
from image_token.validate import (
    check_if_path_is_file,
    check_if_path_is_folder,
    check_allowed_extensions,
    is_url,
    is_multiple_urls,
)
from image_token.utils import (
    read_image_dims,
    list_all_images,
    get_image_dimensions_from_bytes,
)

from image_token.config import gemini_config
from tqdm import tqdm
from pathlib import Path
from image_token.caching_utils import ImageDimensionCache
import requests
from requests.exceptions import HTTPError, RequestException
import math


class GeminiModel(VisionModel):

    def __init__(self):
        print("gemini called")


    def process_image(self, path: str, model_name: str = None):
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

        num_tokens = self.calculate_image_tokens(
            model_name=model_name,
            width=width,
            height=height,
        )

        return num_tokens

    def process_image_from_url(
        self, url: str, model_name: str = None, cache: ImageDimensionCache = None
    ) -> int:
        """
        Process an image from a URL and calculate number of tokens, using persistent dimension cache.

        """
        try:
            dimensions = cache.get_cached_dimensions(url)

            if dimensions:
                width, height = dimensions
            else:
                response = requests.get(url)
                response.raise_for_status()

                image_bytes = response.content
                width, height = get_image_dimensions_from_bytes(image_bytes)
                cache.cache_dimensions(url, width, height)

            num_tokens = self.calculate_image_tokens(
                model_name=model_name,
                width=width,
                height=height,
            )
            return num_tokens
        except HTTPError as http_err:
            print(f"HTTP error occurred while fetching image: {http_err}")
        except RequestException as req_err:
            print(f"Network error occurred: {req_err}")
        except Exception as err:
            print(f"An unexpected error occurred: {err}")

        return -1

    def get_token(self, model_name, path, save_to=None, **kwargs):

        """
        Calculate and return the total number of tokens for a given image or directory of images.

        This function processes an image or a directory of images, calculates the number of tokens
        for each image based on the given model configuration, and optionally saves the results to
        a file. The total number of tokens includes the prefix tokens.Prefix is not used for gemini but just to keep
        codebase consistent

        Args:
            model_name (str): The name of the model to use for token calculation.
            path (str): The path to the image file or directory containing images.
            prefix_tokens (int, optional): The number of tokens to add as a prefix. Defaults to 9.
            save_to (str, optional): The path to a file where the results should be saved. If None,
                                    the results are not saved to a file.

        Returns:
            int: The total number of tokens calculated.
        """
        print("gettokne : " + model_name)
        result_dict = {}
        total_tokens = 0

        if check_if_path_is_file(path=path):
            num_tokens = self.process_image(path=path, model_name=model_name)
            total_tokens = num_tokens
            result_dict[path] = total_tokens
        elif check_if_path_is_folder(path=path):
            image_files = list_all_images(path=path)
            total_tokens = 0
            for image_path in tqdm(image_files):
                num_tokens = self.process_image(path=image_path, model_name=model_name)
                total_tokens += num_tokens
                result_dict[image_path] = num_tokens
        elif is_url(path=path):
            with ImageDimensionCache() as cache:
                num_tokens = self.process_image_from_url(
                    url=path, model_name=model_name, cache=cache
                )
            total_tokens = num_tokens
            result_dict[path] = total_tokens
        elif is_multiple_urls(urls=path):
            with ImageDimensionCache() as cache:
                for url in path:
                    num_tokens = self.process_image_from_url(
                        url=url, model_name=model_name, cache=cache
                    )
                    total_tokens += num_tokens
                    result_dict[url] = num_tokens
        else:
            raise ValueError(
                f"Invalid input path or URL: '{path}'. The given input is not a valid file, folder, or URL."
            )

        if save_to:
            with open(save_to, "w") as f:
                json.dump(result_dict, f, indent=4)

        return total_tokens

    def calculate_image_tokens(self, model_name: str, width: int, height: int):
        """Calculate the number of image tokens for Gemini models.

        This function calculates the number of image tokens required for Gemini models based on the image dimensions and model version.

        Args:
            model_name (str): The name of the Gemini model (e.g., "gemini-1.5-pro").
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            int: The estimated number of image tokens.

        Notes:
            - For Gemini models other than version 2.0, a base of 258 tokens is added.
            - For Gemini 2.0:
                - If both width and height are <= 384, a base of 258 tokens is added.
                - Otherwise, the image is divided into tiles, and 258 tokens are added per tile.
                - The tile size is determined by the smaller side of the image, with a minimum of 256 and a maximum of 768.
        """

        num_tokens = 0
        model_version = model_name.split("-")[1]
        if model_version != "2.0":
            num_tokens += 258
        else:
            if width <= 384 and height <= 384:
                num_tokens += 258

            smaller_side = min(width, height)
            tile_size = min(max(smaller_side / 1.5, 256), 768)

            tiles_w = math.ceil(width / tile_size)
            tiles_h = math.ceil(height / tile_size)

            total_tiles = tiles_w * tiles_h + 1
            num_tokens += total_tiles * 258

        return num_tokens

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        config: dict,
        input_modality: str = "text",
    ) -> float:
        """
        Calculate the estimated cost of a request to a Gemini model with dynamic pricing.

        Args:
            input_tokens (int): The number of tokens in the input.
            output_tokens (int): The number of tokens in the output.
            config (dict): The configuration dictionary for the model.
            input_modality (str): The modality of the input ('text', 'image', 'video', 'audio').
                                Defaults to 'text'.

        Returns:
            float: The estimated cost in dollars.
        """
        tier = None
        for t in config.get("pricing_tiers", []):
            if input_tokens <= t["up_to_tokens"]:
                tier = t
                break

        if not tier:
            raise ValueError(
                "No suitable pricing tier found for the given number of input tokens."
            )

        input_cost_rate = tier.get("input_cost_per_million_tokens", 0)
        if isinstance(input_cost_rate, dict):
            input_cost_rate = input_cost_rate.get(input_modality, 0)

        output_cost_rate = tier.get("output_cost_per_million_tokens", 0)

        input_cost = (input_tokens / 1_000_000) * input_cost_rate
        output_cost = (output_tokens / 1_000_000) * output_cost_rate

        return input_cost + output_cost

    def get_cost(
        self,
        model_name: str,
        system_prompt_tokens: int,
        approx_output_tokens: int,
        path: Path | str,
        save_to: str = None,
        input_modality: str = "text",
        **kwargs
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
            input_modality (str): The modality of the input for Gemini models.

        Returns:
            float: The estimated cost in dollars.
        """
        print(model_name)
        total_input_tokens = system_prompt_tokens + self.get_token(
            model_name=model_name,
            path=path,
            save_to=save_to,
        )

        model_config = gemini_config.get(model_name)
        if not model_config:
            raise ValueError(f"Configuration for model '{model_name}' not found.")

        cost = self.calculate_cost(
            input_tokens=total_input_tokens,
            output_tokens=approx_output_tokens,
            config=model_config,
            input_modality=input_modality,
        )
        return cost
