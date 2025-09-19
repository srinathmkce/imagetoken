import pytest
import tempfile
import os
import json
from PIL import Image
from tempfile import NamedTemporaryFile
from image_token import get_token
from pathlib import Path
from conftest import (
    JPG_FILE_PATH,
    JPEG_FILE_PATH,
    PNG_FILE_PATH,
    JPG_URL,
    JPEG_URL,
    PNG_URL,
    CACHE_TEST_IMAGE_URL,
    GPT_4_1_MINI_MODEL_NAME,
    GPT_4_1_NANO_MODEL_NAME,
    GPT_MODEL_NAMES,
    EXPECTED_OUTPUT_TOKENS_GPT,
    GEMINI_MODEL_NAMES,
    EXPECTED_OUTPUT_TOKENS_GEMINI,
    test_cases,
    test_inputs,
)
from image_token.config import openai_config
from image_token.caching_utils import ImageDimensionCache
import time
from image_token.validate import (
    check_if_path_is_file,
    check_if_path_is_folder,
    is_url,
    is_multiple_urls,
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


@pytest.mark.parametrize("model_name", GPT_MODEL_NAMES)
def test_valid_file_path(model_name):
    assert (
        get_token(model_name=model_name, path=JPG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS_GPT[model_name][JPG_FILE_PATH]
    )


@pytest.mark.parametrize("model_name", GPT_MODEL_NAMES)
def test_multiple_fomat(model_name):
    assert (
        get_token(model_name=model_name, path=JPG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS_GPT[model_name][JPG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=JPEG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS_GPT[model_name][JPEG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=PNG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS_GPT[model_name][PNG_FILE_PATH]
    )


@pytest.mark.parametrize("model_name", GPT_MODEL_NAMES)
def test_multiple_fomat_url(model_name):
    assert (
        get_token(model_name=model_name, path=JPG_URL)
        == EXPECTED_OUTPUT_TOKENS_GPT[model_name][JPG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=JPEG_URL)
        == EXPECTED_OUTPUT_TOKENS_GPT[model_name][JPEG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=PNG_URL)
        == EXPECTED_OUTPUT_TOKENS_GPT[model_name][PNG_FILE_PATH]
    )


@pytest.mark.parametrize("model_name", GEMINI_MODEL_NAMES)
def test_valid_file_path(model_name):
    assert (
        get_token(model_name=model_name, path=JPG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS_GEMINI[model_name][JPG_FILE_PATH]
    )


@pytest.mark.parametrize("model_name", GEMINI_MODEL_NAMES)
def test_multiple_fomat(model_name):
    assert (
        get_token(model_name=model_name, path=JPG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS_GEMINI[model_name][JPG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=JPEG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS_GEMINI[model_name][JPEG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=PNG_FILE_PATH)
        == EXPECTED_OUTPUT_TOKENS_GEMINI[model_name][PNG_FILE_PATH]
    )


@pytest.mark.parametrize("model_name", GEMINI_MODEL_NAMES)
def test_multiple_fomat_url(model_name):
    assert (
        get_token(model_name=model_name, path=JPG_URL)
        == EXPECTED_OUTPUT_TOKENS_GEMINI[model_name][JPG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=JPEG_URL)
        == EXPECTED_OUTPUT_TOKENS_GEMINI[model_name][JPEG_FILE_PATH]
    )
    assert (
        get_token(model_name=model_name, path=PNG_URL)
        == EXPECTED_OUTPUT_TOKENS_GEMINI[model_name][PNG_FILE_PATH]
    )


@pytest.mark.parametrize("model_name", GPT_MODEL_NAMES)
def test_get_tokens_with_folder(model_name):
    path = str(Path("tests") / "image_folder")
    assert (
        get_token(model_name=model_name, path=path)
        == EXPECTED_OUTPUT_TOKENS_GPT[model_name]["total_tokens"]
    )


@pytest.mark.parametrize("model_name", GPT_MODEL_NAMES)
def test_get_tokens_for_file_and_save(model_name):
    output_path = f"{model_name}_output.json"
    expected_tokens = EXPECTED_OUTPUT_TOKENS_GPT[model_name][JPG_FILE_PATH]
    assert (
        get_token(model_name=model_name, path=JPG_FILE_PATH, save_to=output_path)
        == expected_tokens
    )

    with open(output_path, "r") as f:
        result = json.load(f)
        assert len(result) == 1
        assert result[JPG_FILE_PATH] == expected_tokens

    os.remove(output_path)


@pytest.mark.parametrize("model_name", GPT_MODEL_NAMES)
def test_get_tokens_for_folder_and_save(model_name):
    path = str(Path("tests") / "image_folder")
    output_path = f"{model_name}_output.json"
    total_tokens = EXPECTED_OUTPUT_TOKENS_GPT[model_name]["total_tokens"]
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


def test_sqlite_based_image_cache():
    config = openai_config["gpt-4.1-mini"]

    with ImageDimensionCache() as cache_instance:
        cache_instance.delete_dimensions(CACHE_TEST_IMAGE_URL)
        assert cache_instance.get_cached_dimensions(CACHE_TEST_IMAGE_URL) is None

        start_time = time.time()
        tokens_1 = get_token(
            path=CACHE_TEST_IMAGE_URL,
            model_name="gpt-4.1-mini",
        )
        duration_1 = time.time() - start_time
        print(f"[First call] Duration without cache: {duration_1:.6f} seconds")

        assert tokens_1 > 0
        assert cache_instance.get_cached_dimensions(CACHE_TEST_IMAGE_URL) is not None

        start_time = time.time()
        tokens_2 = get_token(
            path=CACHE_TEST_IMAGE_URL,
            model_name="gpt-4.1-mini",
        )
        duration_2 = time.time() - start_time
        print(f"[Second call] Duration with cache: {duration_2:.10f} seconds")

        assert tokens_2 == tokens_1
        assert duration_2 < duration_1

        cache_instance.delete_dimensions(CACHE_TEST_IMAGE_URL)
        assert cache_instance.get_cached_dimensions(CACHE_TEST_IMAGE_URL) is None

        start_time = time.time()
        tokens_3 = get_token(
            path = CACHE_TEST_IMAGE_URL,
            model_name="gpt-4.1-mini",
        )
        duration_3 = time.time() - start_time
        print(f"[Third call after cache delete] Duration: {duration_3:.6f} seconds")

        assert tokens_3 == tokens_1
        assert cache_instance.get_cached_dimensions(CACHE_TEST_IMAGE_URL) is not None


def test_check_if_path_is_file():
    """Test check_if_path_is_file - only file should return True"""
    assert check_if_path_is_file(test_inputs["file"]) == True

    for key in test_inputs:
        if key != "file":
            assert check_if_path_is_file(test_inputs[key]) == False


def test_check_if_path_is_folder():
    """Test check_if_path_is_folder - only folder should return True"""
    assert check_if_path_is_folder(test_inputs["folder"]) == True

    for key in test_inputs:
        if key != "folder":
            assert check_if_path_is_folder(test_inputs[key]) == False


def test_is_url():
    """Test is_url - only URL should return True"""
    assert is_url(test_inputs["url"]) == True

    for key in test_inputs:
        if key != "url":
            assert is_url(test_inputs[key]) == False


def test_is_multiple_urls():
    """Test is_multiple_urls - only URLs list should return True"""
    assert is_multiple_urls(test_inputs["urls"]) == True

    for key in test_inputs:
        if key != "urls":
            assert is_multiple_urls(test_inputs[key]) == False
