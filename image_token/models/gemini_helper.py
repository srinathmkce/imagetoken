from image_token.base.base import VisionModel
from image_token.utils.config import gemini_config
import math


class GeminiModel(VisionModel):

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
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate the estimated cost of a request to a Gemini model with dynamic pricing.

        Args:
            model_name (str): The name of the model.
            input_tokens (int): The number of tokens in the input.
            output_tokens (int): The number of tokens in the output.
    
        Returns:
            float: The estimated cost in dollars.
        """
        model_config = gemini_config[model_name]
        tier = None
        for t in model_config["pricing_tiers"]:
            if input_tokens <= t["up_to_tokens"]:
                tier = t
                break

        if not tier:
            raise ValueError(
                "No suitable pricing tier found for the given number of input tokens."
            )

        input_cost_rate = tier.get("input_cost_per_million_tokens", 0)


        output_cost_rate = tier.get("output_cost_per_million_tokens", 0)

        input_cost = (input_tokens / 1000000) * input_cost_rate
        output_cost = (output_tokens / 1000000) * output_cost_rate

        return input_cost + output_cost
