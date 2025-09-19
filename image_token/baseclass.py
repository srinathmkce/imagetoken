from abc import ABC, abstractmethod
from pathlib import Path
import json
import tqdm
import requests
from requests.exceptions import HTTPError, RequestException
from image_token.validate import (
    check_if_path_is_file,
    check_if_path_is_folder,
    is_url,
    is_multiple_urls,
    check_allowed_extensions,
)
from image_token.utils import (
    list_all_images,
    read_image_dims,
    get_image_dimensions_from_bytes
)
from image_token.caching_utils import ImageDimensionCache

class VisionModel(ABC):
    
    def process_image(self, path: str, model_name: str = None , **kwargs):
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
            **kwargs
        )

        return num_tokens

    def process_image_from_url(
        self, url: str, model_name: str = None, cache: ImageDimensionCache = None , **kwargs
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
                **kwargs
            )
            return num_tokens
        except HTTPError as http_err:
            print(f"HTTP error occurred while fetching image: {http_err}")
        except RequestException as req_err:
            print(f"Network error occurred: {req_err}")
        except Exception as err:
            print(f"An unexpected error occurred: {err}")

        return -1

    @abstractmethod
    def calculate_image_tokens(self, name: str, h: int, w: int, config: dict):
        """Calculate token count based on image dimensions and configuration."""
        pass

    def get_token(self, model_name, path, save_to=None, **kwargs):
        result_dict = {}
        total_tokens = 0

        if check_if_path_is_file(path=path):
            num_tokens = self.process_image(path=path, model_name=model_name, **kwargs)
            total_tokens = num_tokens
            result_dict[str(path)] = total_tokens

        elif check_if_path_is_folder(path=path):
            image_files = list_all_images(path=path)
            for image_path in tqdm.tqdm(image_files):
                num_tokens = self.process_image(path=image_path, model_name=model_name, **kwargs)
                total_tokens += num_tokens
                result_dict[str(image_path)] = num_tokens

        elif is_url(path=path):
            with ImageDimensionCache() as cache:
                num_tokens = self.process_image_from_url(url=path, model_name=model_name, cache=cache, **kwargs)
            total_tokens = num_tokens
            result_dict[path] = total_tokens

        elif is_multiple_urls(urls=path):
            with ImageDimensionCache() as cache:
                for url in path:
                    num_tokens = self.process_image_from_url(url=url, model_name=model_name, cache=cache, **kwargs)
                    total_tokens += num_tokens
                    result_dict[url] = num_tokens

        else:
            raise ValueError(f"Invalid input path or URL: '{path}'.")

        if save_to:
            with open(save_to, "w") as f:
                json.dump(result_dict, f, indent=4)

        return total_tokens



    @abstractmethod
    def calculate_cost(self, input_token: int, ouput_tokens: int, config: dict):
        "Estimate the cost of"
        pass

    def get_cost(
        self,
        model_name: str,
        system_prompt_tokens: int,
        approx_output_tokens: int,
        path: Path | str,
        save_to: str = None,
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
        total_input_tokens = system_prompt_tokens + self.get_token(
            model_name=model_name,
            path=path,
            save_to=save_to,
        )

        cost = self.calculate_cost(
            model_name=model_name,
            input_tokens=total_input_tokens,
            output_tokens=approx_output_tokens,
        )
        return cost
