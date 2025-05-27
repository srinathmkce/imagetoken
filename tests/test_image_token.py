import pytest
import tempfile
import os
import json
from image_token import get_token
from pathlib import Path


def test_invalid_file_path():
    with pytest.raises(FileNotFoundError):
        get_token(model_name="gpt-4.1-nano", path=r"dummy.png")


def test_invalid_model_name():
    with pytest.raises(ValueError):
        get_token(model_name="dummy", path = str(Path("tests") / "image_folder" / "kitten.jpg"))


def test_invalid_file_extension():
    # create a temp folder and write a txt file in it and remove it at the end
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = f"{temp_dir}/dummy.txt"
        with open(temp_file_path, "w") as f:
            f.write("This is a dummy file")

        with pytest.raises(ValueError):
            get_token(model_name="gpt-4.1-mini", path=temp_file_path)


def test_valid_file_path():
    assert (
        get_token(model_name="gpt-4.1-mini", path = str(Path("tests") / "image_folder" / "kitten.jpg"))
        == 865
    )


def test_multiple_fomat():
    assert (
        get_token(model_name="gpt-4.1-mini", path = str(Path("tests") / "image_folder" / "kitten.jpg"))
        == 865
    )
    assert (
        get_token(model_name="gpt-4.1-mini", path = str(Path("tests") / "image_folder" / "kitten.jpeg"))
        == 2473
    )
    assert (
        get_token(model_name="gpt-4.1-mini", path = str(Path("tests") / "image_folder" / "kitten.png"))
        == 423
    )


def test_get_tokens_with_folder():
    path = str(Path("tests") / "image_folder")
    assert get_token(model_name="gpt-4.1-mini", path=path) == 3761


def test_get_tokens_for_file_and_save():
    path = str(Path("tests") / "image_folder" / "kitten.jpg")
    output_path = "output.json"
    assert get_token(model_name="gpt-4.1-mini", path=path, save_to=output_path) == 865

    with open(output_path, "r") as f:
        result = json.load(f)
        assert len(result) == 1
        assert result[path] == 865

    os.remove(output_path)


def test_get_tokens_for_folder_and_save():
    path = str(Path("tests") / "image_folder")
    output_path = "output.json"
    assert get_token(model_name="gpt-4.1-mini", path=path, save_to=output_path) == 3761

    with open(output_path, "r") as f:
        result = json.load(f)
        assert len(result) == 3
    os.remove(output_path)


def test_multiple_models():
    assert (
        get_token(model_name="gpt-4.1-mini", path = str(Path("tests") / "image_folder" / "kitten.jpg"))
        == 865
    )
    assert (
        get_token(model_name="gpt-4.1-nano", path = str(Path("tests") / "image_folder" / "kitten.jpg"))
        == 1310
    )
