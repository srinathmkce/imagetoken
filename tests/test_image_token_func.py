import base64
import os
import pytest
from image_token import get_token
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found in .env"


def test_with_openai_api_key():
    image_path = "tests/kitten.jpg"

    calculated_input_tokens = get_token(model_name="gpt-4.1-mini", path=image_path)
    print(
        "Number of input tokens calculated by image_token: ",
        calculated_input_tokens + 9,
    )

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    messages = [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        )
    ]
    response = llm.invoke(messages)
    input_tokens_from_openai = response.usage_metadata["input_tokens"]

    print("Number of input tokens from OpenAI: ", input_tokens_from_openai)

    assert int(calculated_input_tokens + 9) == input_tokens_from_openai


@pytest.mark.parametrize("model_name", ["gpt-4.1-mini", "gpt-4.1-nano"])
def test_with_openai_api_key_models(model_name):
    image_path = "tests/kitten.jpg"

    calculated_input_tokens = get_token(model_name=model_name, path=image_path)
    print(
        "Number of input tokens calculated by image_token: ",
        calculated_input_tokens + 9,
    )

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    llm = ChatOpenAI(
        model=model_name, temperature=0, max_tokens=None, timeout=None, max_retries=2
    )

    messages = [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        )
    ]
    response = llm.invoke(messages)
    input_tokens_from_openai = response.usage_metadata["input_tokens"]

    print("Number of input tokens from OpenAI: ", input_tokens_from_openai)

    assert int(calculated_input_tokens + 9) == input_tokens_from_openai
