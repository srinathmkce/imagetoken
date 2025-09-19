from image_token.baseclass import VisionModel
from image_token.config import openai_config
import math
from image_token.config import patch_models, tile_models


class OpenAiModel(VisionModel):

    def calculate_image_tokens(
        self,
        model_name: str,
        width: int,
        height: int,
        **kwargs
    ):
        """
        Calculate the number of tokens for an image based on the model configuration.

        Args:
            model_name (str): The name of the model.
            width (int): The width of the image.
            height (int): The height of the image.
            max_tokens (int): The maximum number of tokens for the model.
            model_config (dict): The configuration for the model.

        Returns:
            int: The number of tokens for the image.
        """
        prefix_tokens = kwargs.get("prefix_tokens", 9)

        if model_name in patch_models:
            model_config = openai_config[model_name]
            max_tokens = model_config[max_tokens]
            num_tokens = self.calculate_image_tokens_patch(
                width=width, height=height, max_tokens=max_tokens
            )
            return int(num_tokens * model_config["factor"]) + prefix_tokens

        elif model_name in tile_models:
            num_tokens = self.calculate_image_tokens_tile(
                width=width,
                height=height,
                tile_models=tile_models,
                model_name=model_name,
            )
            return num_tokens + prefix_tokens
        else:
            raise ValueError(
                f"Model {model_name} is not supported for image token calculation."
            )

    def calculate_image_tokens_patch(
        self, width, height, max_tokens=1536, patch_size=32
    ):
        # Step 1: Calculate number of patches without scaling
        """
        Calculates the number of image tokens based on the dimensions of the image.

        This function determines the number of patches that can be extracted from an
        image with given width and height. It first calculates the total number of
        patches without any scaling, and if the total number of patches exceeds the
        maximum allowed tokens, it scales down the image dimensions to fit within
        the maximum token limit.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            patch_size (int, optional): The size of each patch. Defaults to 32.
            max_tokens (int, optional): The maximum number of tokens allowed.
                                        Defaults to 1536.

        Returns:
            int: The calculated number of image tokens.
        """

        patches_w = (width + patch_size - 1) // patch_size
        patches_h = (height + patch_size - 1) // patch_size
        total_patches = patches_w * patches_h

        if total_patches <= max_tokens:
            return total_patches

        # Step 2: Scaling required
        # Calculate shrink factor using given formula
        shrink_factor = math.sqrt((max_tokens * patch_size**2) / (width * height))
        scaled_width = width * shrink_factor
        scaled_height = height * shrink_factor

        # Calculate patch count from scaled dimensions
        patches_w_scaled = scaled_width / patch_size
        patches_h_scaled = scaled_height / patch_size

        # Fit the image to an integer number of patches (round down width)
        adjusted_patches_w = int(patches_w_scaled)
        adjusted_patches_h = int(patches_h_scaled)
        scale_adjust_w = adjusted_patches_w / patches_w_scaled
        scale_adjust_h = adjusted_patches_h / patches_h_scaled

        final_width = scaled_width * scale_adjust_w
        final_height = scaled_height * scale_adjust_h

        final_patches_w = int(final_width / patch_size)
        final_patches_h = int(final_height / patch_size)

        # Final token count is number of patches
        return final_patches_w * final_patches_h

    def calculate_image_tokens_tile(
        self, width, height, tile_models, model_name, tile_size=512
    ):
        """Calculate the number of image tokens.m

        The calculation is based on the dimensions of the image and the specified tile model.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            tile_models (dict): A dictionary containing tile model configurations.
            model_name (str): The name of the model to use for token calculation.
            tile_size (int, optional): The size of each tile. Defaults to 512.

        Returns:
            int: The calculated number of image tokens.

        Notes:
            - The function first scales the image to fit within a 2048x2048 box.
            - It then resizes the shortest side to 768 pixels.
            - Finally, it counts how many 512x512 tiles fit into the resized image.
            - The number of tokens is calculated based on the base token count and the number of tiles.
        """

        # Step 1: Scale to fit in 2048x2048
        max_side = 2048
        width_resized = width
        height_resized = height
        if width > max_side or height > max_side:
            scale_factor = max_side / max(width, height)
            width_resized = width * scale_factor
            height_resized = height * scale_factor

        # Step 2: Resize shortest side to 768
        final_width = width_resized
        final_height = height_resized
        shortest = min(width_resized, height_resized)
        if shortest > 768:
            scale_factor_2 = 768 / shortest
            final_width = width_resized * scale_factor_2
            final_height = height_resized * scale_factor_2

        # Step 3: Count tiles (each 512x512)
        tiles_w = math.ceil(final_width / tile_size)
        tiles_h = math.ceil(final_height / tile_size)
        tile_count = tiles_w * tiles_h

        base = tile_models[model_name]["base"]
        per_tile = tile_models[model_name]["tile"]
        return base + tile_count * per_tile

    def calculate_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
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
        model_config = openai_config[model_name]
        input_cost = (input_tokens / 10**6) * model_config["input_tokens"]
        output_cost = (output_tokens / 10**6) * model_config["output_tokens"]
        return input_cost + output_cost
