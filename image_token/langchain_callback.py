from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
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
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

    def _get_model_name(self, kwargs):
        # check invocation_params is in kwargs
        if "invocation_params" in kwargs:
            # check model_name is in invocation_params
            if "model_name" in kwargs["invocation_params"]:
                return kwargs["invocation_params"]["model_name"]
        return None

    def _get_model_config(self, model_name):
        if model_name in openai_config:
            return openai_config[model_name]
        return None

    def _get_image_size_from_data_url(self, data_url):
        try:
            image_bytes = base64.b64decode(data_url.split(",")[1])
            image = Image.open(BytesIO(image_bytes))
            return image.size
        except Exception as e:
            return None

    def _calculate_image_tokens(self, model_name, width, height):
        model_config = self._get_model_config(model_name)
        if not model_config:
            return None

        num_tokens = calculate_image_tokens(
            model_name=model_name,
            width=width,
            height=height,
            max_tokens=model_config["max_tokens"],
            model_config=model_config,
        )
        return num_tokens

    def _calculate_approx_input_cost(self, model_name, input_tokens):
        model_config = self._get_model_config(model_name)
        if not model_config:
            return None

        approx_cost = calculate_cost(
            input_tokens=input_tokens, output_tokens=0, config=model_config
        )
        return approx_cost

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        model_name = self._get_model_name(kwargs)
        if not model_name:
            return

        for em in messages:
            for message in em:
                if isinstance(message, HumanMessage):
                    message_content = message.content
                    for content in message_content:
                        if isinstance(content, dict) and "image_url" in content:
                            image_url_dict = content["image_url"]
                            if (
                                isinstance(image_url_dict, dict)
                                and "url" in image_url_dict
                            ):
                                image_url = image_url_dict["url"]

                                if image_url.startswith("data:image"):
                                    image_size = self._get_image_size_from_data_url(
                                        image_url
                                    )
                                    if image_size:
                                        width, height = image_size
                                        # print(f"Width: {width}, Height: {height}")
                                        num_tokens = self._calculate_image_tokens(
                                            model_name, width, height
                                        )
                                        approx_input_cost = (
                                            self._calculate_approx_input_cost(
                                                model_name, num_tokens
                                            )
                                        )
                                        if self.verbose:
                                            print(
                                                "Number of input tokens: ", num_tokens
                                            )
                                            print(
                                                f"Approximate input cost: ${approx_input_cost:.4f}"
                                            )


callbacks = [LoggingHandler(verbose=False)]
llm = ChatOpenAI(model="gpt-4.1-nano")

path = JPG_FILE_PATH = str(Path("tests") / "image_folder" / "kitten.jpg")

# response = llm.invoke("What is the capital of France?", config={"callbacks": callbacks})
with open(JPG_FILE_PATH, "rb") as image_file:
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
response = llm.invoke(messages, config={"callbacks": callbacks})

print(response)

# prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")

# chain = prompt | llm

# chain_with_callbacks = chain.with_config(callbacks=callbacks)

# chain_with_callbacks.invoke({"number": "2"})
