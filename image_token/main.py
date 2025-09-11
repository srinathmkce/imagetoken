from abc import ABC, abstractmethod
from pathlib import Path


class VisionModel(ABC):
    @abstractmethod
    def get_token(self, *args, **kwargs):
        """Retrieve a token based on name and path."""
        pass

    @abstractmethod
    def process_image(self, path: str, name: str, config: dict):
        """Process an image from a local path using the given configuration."""
        pass

    @abstractmethod
    def process_image_from_url(self, url: str, name: str, config: dict):
        """Process an image from a URL using the given configuration."""
        pass

    @abstractmethod
    def calculate_image_tokens(self, name: str, h: int, w: int, config: dict):
        """Calculate token count based on image dimensions and configuration."""
        pass

    @abstractmethod
    def get_cost(
        self, *args, **kwargs
    ):
        """Estimate the cost based on prompt and token usage."""
        pass

    @abstractmethod
    def calculate_cost(self, input_token: int, ouput_tokens: int, config: dict):
        "Estimate the cost of"
        pass
