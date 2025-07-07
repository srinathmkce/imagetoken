import json
from image_token.validate import (
    check_if_file_or_folder_exists,
    check_if_path_is_file,
    check_allowed_extensions,
    check_valid_model,
)
from image_token.utils import read_image_dims, list_all_images
from image_token.core import calculate_image_tokens
from image_token.config import openai_config
from tqdm import tqdm


class GetTokens:
    """Calculate the number of tokens in an image or directory of images.

    Args:
        model_name (str): The name of the model to use for token calculation.
        path (str): The path to the image file or directory containing images.
        prefix_tokens (int, optional): The number of tokens to add as a prefix. Defaults to 9.
        save_to (str, optional): The path to a file where the results should be saved. If None, results are not saved.

    Returns:
        int: The total number of tokens calculated from the image(s).
    """
    def __init__(self, model_name: str, path: str, prefix_tokens: int = 9, save_to: str = None):
        check_if_file_or_folder_exists(path)
        check_valid_model(model_name)

        self.model_name = model_name
        self.path = path
        self.prefix_tokens = prefix_tokens
        self.save_to = save_to
        self.model_config = openai_config[model_name]

    def _process_image(self, path: str) -> int:
        """
        Process an image and calculate the number of tokens in it.

        Args:
            path (str): The path to the image file.

        Returns:
            int: The number of tokens in the image.
        """
        check_allowed_extensions(path)
        width, height = read_image_dims(path)
        max_tokens = self.model_config["max_tokens"]
        return calculate_image_tokens(
            model_name=self.model_name,
            width=width,
            height=height,
            max_tokens=max_tokens,
            model_config=self.model_config,
        )

    def get_token_sync(self) -> int:
        """Calculates number of tokens in an image or directory of images.
        
        It processes each image, calculates the number of tokens, and returns the total count.
        It also saves the results to a JSON file if `save_to` is specified.
        This function is synchronous and can be used in a blocking context.

        Args:
            None

        Returns:
            int: The total number of tokens in the image(s).
        
        Notes:
            - If the path is a file, it processes that file.
            - If the path is a directory, it processes all images in that directory.
            - It saves the results to a JSON file if `save_to` is specified.
        """

        result_dict = {}
        total_tokens = 0

        if check_if_path_is_file(self.path):
            num_tokens = self._process_image(self.path)
            total_tokens = num_tokens + self.prefix_tokens
            result_dict[self.path] = total_tokens
        else:
            image_files = list_all_images(self.path)
            for image_path in tqdm(image_files):
                num_tokens = self._process_image(image_path)
                total_tokens += num_tokens + self.prefix_tokens
                result_dict[image_path] = num_tokens + self.prefix_tokens

        if self.save_to:
            with open(self.save_to, "w") as f:
                json.dump(result_dict, f, indent=4)

        return total_tokens

    async def get_token_async(self) -> int:
        """Asynchronous version of get_token_sync.

        This function is intended to be used in an asynchronous context.
        It calls the synchronous version of the token calculation method.
        It does not perform any asynchronous operations itself but is structured
        to fit into an async workflow.

        Returns:
            int: The total number of tokens in the image(s).
        """
        # Same logic as get_token_sync but inside async context
        return self.get_token_sync()  # or make this logic truly async