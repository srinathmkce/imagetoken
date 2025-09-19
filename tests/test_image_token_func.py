import base64
import os
import PIL.Image
import pytest
from image_token import get_token
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from conftest import GPT_MODEL_NAMES, GEMINI_MODEL_NAMES, JPG_FILE_PATH, JPG_URL
import requests
import PIL
from google import genai
from google.genai import types

@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found in .env"

@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
    assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not found in .env"


@pytest.mark.parametrize("model_name", GPT_MODEL_NAMES)
def test_with_openai_api_key_models(model_name):
    """Test the image token calculation with OpenAI API key models.

    This test checks if the number of input tokens calculated by the image_token
    library matches the number of input tokens returned by the OpenAI API for
    the specified model when processing an image.

    Args:
        model_name (str): The name of the model to test.

    Notes:
        - The test reads a JPEG image file, encodes it to base64, and sends it
          to the OpenAI API.
        - It compares the number of input tokens calculated by the image_token
          library with the number of input tokens returned by the OpenAI API.
        - The test requires an OpenAI API key to be set in the environment.
    """
    # calculate input tokens based on the model_name
    calculated_input_tokens = get_token(model_name=model_name, path=JPG_FILE_PATH)
    print(
        "Number of input tokens calculated by image_token: ",
        calculated_input_tokens,
    )

    with open(JPG_FILE_PATH, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    llm = ChatOpenAI(model=model_name, max_tokens=None, timeout=None, max_retries=2)

    messages = [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        )
    ]
    response = llm.invoke(messages)
    print(response)
    input_tokens_from_openai = response.usage_metadata["input_tokens"]

    print("Number of input tokens from OpenAI: ", input_tokens_from_openai)

    assert int(calculated_input_tokens) == pytest.approx(
        input_tokens_from_openai, abs=6
    )


@pytest.mark.parametrize("model_name", GPT_MODEL_NAMES)
def test_process_image_from_url_with_openai_api_integration(model_name):
    """
    Integration test comparing process_image_from_url with OpenAI API results.

    This test downloads an image from a URL, processes it with your function,
    and compares the token count with the actual OpenAI API response.

    Args:
        model_name (str): The name of the model to test.

    Notes:
        - This test makes actual HTTP requests to download images
        - It requires an OpenAI API key to be set in the environment
        - The test compares calculated tokens with actual OpenAI API response
    """

    calculated_input_tokens = get_token(model_name=model_name, path=JPG_URL)
    print(
        "Number of input tokens calculated by image_token: ",
        calculated_input_tokens,
    )

    response = requests.get(JPG_URL)
    response.raise_for_status()
    image_bytes = response.content
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    llm = ChatOpenAI(model=model_name, max_tokens=None, timeout=None, max_retries=2)
    messages = [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        )
    ]

    api_response = llm.invoke(messages)

    input_tokens_from_openai = api_response.usage_metadata["input_tokens"]
    print(f"Input tokens from OpenAI API: {input_tokens_from_openai}")

    assert int(calculated_input_tokens) == pytest.approx(
        input_tokens_from_openai, abs=6
    )



@pytest.mark.parametrize("model_name", GEMINI_MODEL_NAMES)
def test_with_gemini_api_key_models(model_name):
    """Test the image token calculation with Gemini API key models.

    This test checks if the number of input tokens calculated by the image_token
    library matches the number of input tokens returned by the Gemini API for
    the specified model when processing an image.

    Args:
        model_name (str): The name of the model to test.

    Notes:
        - The test reads a JPEG image file, encodes it to base64, and sends it
          to the OpenAI API.
        - It compares the number of input tokens calculated by the image_token
          library with the number of input tokens returned by the Gemini API.
        - The test requires an Gemini API key to be set in the environment.
    """
    # calculate input tokens based on the model_name

    calculated_input_tokens = get_token(model_name=model_name, path=JPG_FILE_PATH)
    
    print(
        "Number of input tokens calculated by image_token: ",
        calculated_input_tokens,
    )

    image = PIL.Image.open(JPG_FILE_PATH)
    client = genai.Client()

    response = client.models.generate_content(
        model=model_name,
        contents=[image],
        config=types.GenerateContentConfig(),
    )

    usage = response.usage_metadata
    total_tokens = usage.total_token_count
    prompt_details = usage.prompt_tokens_details or []

    image_input_tokens_from_gemini = 0

    for detail in prompt_details:
        if detail.modality == "IMAGE":
            image_input_tokens_from_gemini += detail.token_count

    print("Number of input tokens from Gemini: ", image_input_tokens_from_gemini)

    assert int(calculated_input_tokens) == pytest.approx(
        image_input_tokens_from_gemini, abs=6
    )
