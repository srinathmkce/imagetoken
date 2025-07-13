from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
from image_token.core import calculate_image_tokens, calculate_text_tokens
from image_token.main import process_image_from_url
from image_token.config import openai_config
from image_token.utils import calculate_cost
from urllib.parse import urlparse
from image_token.caching_utils import ImageDimensionCache


from dotenv import load_dotenv

load_dotenv()


class LoggingHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.total_tokens = 0
        self.total_cost = 0.0
        self.prefix_tokens = 13

    def _get_model_name(self, kwargs):
        return kwargs.get("invocation_params", {}).get("model_name")

    def _get_model_config(self, model_name):
        return openai_config.get(model_name)

    def _get_image_size_from_data_url(self, data_url):
        try:
            image_bytes = base64.b64decode(data_url.split(",")[1])
            image = Image.open(BytesIO(image_bytes))
            return image.size
        except Exception:
            return None

    def _process_image_from_url(self, url, model_name):
        model_config = self._get_model_config(model_name)
        if not model_config:
            return None
        with ImageDimensionCache() as cache:
            return process_image_from_url(url, model_config=model_config, model_name=model_name,cache=cache)

    def _calculate_image_tokens(self, model_name, width, height):
        model_config = self._get_model_config(model_name)
        if not model_config:
            return None
        return calculate_image_tokens(
            model_name, width, height, model_config["max_tokens"], model_config
        )

    def _calculate_approx_input_cost(self, model_name, input_tokens):
        model_config = self._get_model_config(model_name)
        if not model_config:
            return None
        return calculate_cost(
            input_tokens=input_tokens, output_tokens=0, config=model_config
        )

    def on_chat_model_start(self, serialized, messages, **kwargs):
        model_name = self._get_model_name(kwargs)
        if not model_name:
            return

        for em in messages:
            for message in em:
                if isinstance(message, SystemMessage):
                    text = message.content
                    tokens = calculate_text_tokens(model_name=model_name, text=text)
                    self.total_tokens += tokens
                    continue

                if isinstance(message, HumanMessage):
                    for content in message.content:
                        if isinstance(content, dict) and "image_url" in content:
                            image_url = content["image_url"].get("url")
                            if image_url and image_url.startswith("data:image"):
                                size = self._get_image_size_from_data_url(image_url)
                                if size:
                                    width, height = size
                                    tokens = self._calculate_image_tokens(
                                        model_name, width, height
                                    )
                                    cost = self._calculate_approx_input_cost(
                                        model_name, tokens
                                    )

                                    print(
                                        f"[Simulated] Tokens: {tokens}, Cost: ${cost:.4f}"
                                    )

                                    self.total_tokens += tokens
                            elif image_url and image_url.startswith("https"):
                                tokens = self._process_image_from_url(
                                    image_url, model_name
                                )
                                cost = self._calculate_approx_input_cost(
                                    model_name, tokens
                                )
                                print(
                                    f"[Simulated] Tokens: {tokens}, Cost: ${cost:.4f}"
                                )
                                self.total_tokens += tokens

        self.total_tokens += self.prefix_tokens
        self.total_cost = calculate_cost(
            input_tokens=self.total_tokens,
            output_tokens=0,
            config=openai_config[model_name],
        )


def simulate_image_token_cost(llm, messages: list[BaseMessage]):
    """
    Simulate token and cost estimation for images in a message without hitting the OpenAI API.

    Args:
        llm: A LangChain ChatModel (e.g., ChatOpenAI)
        messages: List of LangChain messages (SystemMessage, HumanMessage)
    """
    handler = LoggingHandler()
    model_name = llm.model_name if hasattr(llm, "model_name") else "unknown"

    try:
        # Simulate callback manually to avoid hitting API
        handler.on_chat_model_start(
            {}, [messages], invocation_params={"model_name": model_name}
        )
    except Exception as e:
        # print(f"[Simulate] Tokens: {e.tokens}, Cost: ${e.cost:.4f}")
        return {"tokens": 0, "cost": 0.0}

    return {"tokens": handler.total_tokens, "cost": round(handler.total_cost, 5)}
