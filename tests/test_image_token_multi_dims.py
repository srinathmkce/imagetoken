# Run these test cases only at the end of release
import os
import pytest
from dotenv import load_dotenv
from conftest import (
    test_dimensions, MODEL_NAMES
)

@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found in .env"


# @pytest.mark.parametrize("width,height", test_dimensions)
# @pytest.mark.parametrize("model_name", MODEL_NAMES)
# def test_get_token_matches_openai_response(model_name, width, height):
#     original_image_path = "tests/image_folder/kitten.jpg"
#     assert os.path.exists(original_image_path), "kitten.jpg not found"

#     # Resize the image
#     with Image.open(original_image_path) as img:
#         resized_img = img.resize((width, height))
#         with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
#             resized_img.save(tmp_file.name)
#             tmp_path = tmp_file.name

#     try:
#         # Get tokens from local method
#         calculated_tokens = get_token(model_name=model_name, path=tmp_path)
#         print(
#             f"[{model_name}] ({width}x{height}) - Local token count: {calculated_tokens}"
#         )

#         # Prepare base64 image for OpenAI
#         with open(tmp_path, "rb") as f:
#             image_bytes = f.read()
#         image_base64 = base64.b64encode(image_bytes).decode("utf-8")
#         image_data_url = f"data:image/jpeg;base64,{image_base64}"

#         # Query OpenAI API
#         llm = ChatOpenAI(
#             model=model_name, temperature=0, max_tokens=None, timeout=60, max_retries=2
#         )
#         messages = [
#             HumanMessage(
#                 content=[{"type": "image_url", "image_url": {"url": image_data_url}}]
#             )
#         ]
#         response = llm.invoke(messages)
#         openai_tokens = response.usage_metadata["input_tokens"]
#         print(
#             f"[{model_name}] ({width}x{height}) - OpenAI token count: {openai_tokens}"
#         )

#         # Compare both values
#         assert int(calculated_tokens) == openai_tokens, (
#             f"Token mismatch for {model_name} at {width}x{height} "
#             f"(local: {calculated_tokens}, openai: {openai_tokens})"
#         )
#     finally:
#         os.remove(tmp_path)