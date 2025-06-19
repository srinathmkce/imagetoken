from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from image_token.core import calculate_image_tokens
from image_token.config import openai_config
from image_token.utils import calculate_cost


from dotenv import load_dotenv

load_dotenv()


class LoggingHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.results = []  # List of dicts: {"tokens": int, "cost": float}

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

                                    self.results.append(
                                        {"tokens": tokens, "cost": cost}
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
    except ImageTokenCostException as e:
        # print(f"[Simulate] Tokens: {e.tokens}, Cost: ${e.cost:.4f}")

        return {"tokens": e.tokens, "cost": e.cost}

    if handler.results:
        total_cost = 0
        total_tokens = 0
        for token_result in handler.results:
            total_cost += token_result["cost"]
            total_tokens += token_result["tokens"]

        handler.results.append({"total_tokens": total_tokens, "total_cost": total_cost})
        return handler.results
    else:
        return {"tokens": 0, "cost": 0.0}


def func_test():
    llm = ChatOpenAI(model="gpt-4.1-nano")

    path = str(Path("tests") / "image_folder" / "kitten.jpg")

    # response = llm.invoke("What is the capital of France?", config={"callbacks": callbacks})
    with open(path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        ),
    ]

    result = simulate_image_token_cost(llm, messages)

    print(result)


def func_test_all_images():
    llm = ChatOpenAI(model="gpt-4.1-nano")
    image_folder = Path("tests/image_folder")

    image_files = (
        list(image_folder.glob("*.jpg"))
        + list(image_folder.glob("*.jpeg"))
        + list(image_folder.glob("*.png"))
    )

    for image_path in image_files:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{image_base64}"

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(
                content=[{"type": "image_url", "image_url": {"url": image_data_url}}]
            ),
        ]

    result = simulate_image_token_cost(llm, messages)

    print(result)


def func_test_all_images_one_list():
    llm = ChatOpenAI(model="gpt-4.1-nano")
    image_folder = Path("tests/image_folder")

    image_files = (
        list(image_folder.glob("*.jpg"))
        + list(image_folder.glob("*.jpeg"))
        + list(image_folder.glob("*.png"))
    )

    messages = [
        SystemMessage(content="You are a helpful assistant."),
    ]
    for image_path in image_files:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{image_base64}"

        messages.append(
            HumanMessage(
                content=[{"type": "image_url", "image_url": {"url": image_data_url}}]
            )
        )
    result = simulate_image_token_cost(llm, messages)

    print(result)


if __name__ == "__main__":
    func_test()
    func_test_all_images()
    func_test_all_images_one_list()
