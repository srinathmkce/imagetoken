import pytest
import tempfile
import os
import json
import asyncio
from PIL import Image
from tempfile import NamedTemporaryFile
from image_token import get_token
from pathlib import Path
from conftest import (
    JPG_FILE_PATH,
    JPEG_FILE_PATH,
    PNG_FILE_PATH,
    GPT_4_1_MINI_MODEL_NAME,
    GPT_4_1_NANO_MODEL_NAME,
    MODEL_NAMES,
    EXPECTED_OUTPUT_TOKENS,
    test_cases,
)


def test_invalid_file_path():
    with pytest.raises(FileNotFoundError):
        get_token(model_name=GPT_4_1_NANO_MODEL_NAME, path=r"dummy.png")


def test_invalid_model_name():
    with pytest.raises(ValueError):
        get_token(model_name="dummy", path=JPG_FILE_PATH)


def test_invalid_file_extension():
    # create a temp folder and write a txt file in it and remove it at the end
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = f"{temp_dir}/dummy.txt"
        with open(temp_file_path, "w") as f:
            f.write("This is a dummy file")

        with pytest.raises(ValueError):
            get_token(model_name=GPT_4_1_MINI_MODEL_NAME, path=temp_file_path)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_valid_file_path(model_name):
    assert (
        get_token(model_name=model_name, path=JPG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS[model_name][JPG_FILE_PATH]
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_multiple_fomat(model_name):
    assert (
        get_token(model_name=model_name, path=JPG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS[model_name][JPG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=JPEG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS[model_name][JPEG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=PNG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS[model_name][PNG_FILE_PATH]
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_get_tokens_with_folder(model_name):
    path = str(Path("tests") / "image_folder")
    assert (
        get_token(model_name=model_name, path=path)
        == EXPECTED_OUTPUT_TOKENS[model_name]["total_tokens"]
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_get_tokens_for_file_and_save(model_name):
    output_path = f"{model_name}_output.json"
    expected_tokens = EXPECTED_OUTPUT_TOKENS[model_name][JPG_FILE_PATH]
    assert (
        get_token(model_name=model_name, path=JPG_FILE_PATH, save_to=output_path)
        == expected_tokens
    )

    with open(output_path, "r") as f:
        result = json.load(f)
        assert len(result) == 1
        assert result[JPG_FILE_PATH] == expected_tokens

    os.remove(output_path)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_get_tokens_for_folder_and_save(model_name):
    path = str(Path("tests") / "image_folder")
    output_path = f"{model_name}_output.json"
    total_tokens = EXPECTED_OUTPUT_TOKENS[model_name]["total_tokens"]
    assert (
        get_token(model_name=model_name, path=path, save_to=output_path) == total_tokens
    )

    with open(output_path, "r") as f:
        result = json.load(f)
        assert len(result) == 3
    os.remove(output_path)


@pytest.mark.parametrize("width,height", test_cases.keys())
def test_get_token_on_resized_images(width, height):
    with Image.open(JPG_FILE_PATH) as img:
        resized_img = img.resize((width, height))
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            resized_img.save(tmp.name)
            tmp_path = tmp.name

    try:
        tokens = get_token(GPT_4_1_MINI_MODEL_NAME, tmp_path)
        expected_tokens = test_cases[(width, height)]
        print(f"Testing ({width}x{height}): Got {tokens}, Expected {expected_tokens}")
        assert tokens == expected_tokens, f"Token count mismatch for ({width}x{height})"
    finally:
        os.remove(tmp_path)

@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", MODEL_NAMES)
async def test_get_token_async(model_name):
    """
    Test asynchronous token calculation using get_token with method='async'.
    This ensures the async implementation works as expected.
    """
    tokens = await asyncio.to_thread(
        get_token,
        model_name=model_name,
        path=JPG_FILE_PATH,
        method="async"
    )
    assert isinstance(tokens, int), f"Expected token count to be int, got {type(tokens)}"
    assert tokens > 0, f"Token count should be positive, got {tokens}"
