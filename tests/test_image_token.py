import pytest
import tempfile
import shutil
import os
from image_token import get_token


def test_invalid_file_path():
    with pytest.raises(FileNotFoundError):
        get_token(model_name="gpt-4.1-nano", path=r"dummy.png")


def test_invalid_model_name():
    with pytest.raises(ValueError):
        get_token(model_name="dummy", path=r"tests\kitten.jpg")


def test_invalid_file_extension():
    # create a temp folder and write a txt file in it and remove it at the end
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = f"{temp_dir}\dummy.txt"
        with open(temp_file_path, "w") as f:
            f.write("This is a dummy file")

        with pytest.raises(ValueError):
            get_token(model_name="gpt-4.1-mini", path=temp_file_path)


def test_valid_file_path():
    assert get_token(model_name="gpt-4.1-mini", path=r"tests\kitten.jpg") == 856


def test_multiple_fomat():
    # copy kitten.jpg to kitten.jpeg and kitten.png
    shutil.copyfile(r"tests\kitten.jpg", r"tests\kitten.jpeg")
    shutil.copyfile(r"tests\kitten.jpg", r"tests\kitten.png")

    assert get_token(model_name="gpt-4.1-mini", path=r"tests\kitten.jpg") == 856
    assert get_token(model_name="gpt-4.1-mini", path=r"tests\kitten.jpeg") == 856
    assert get_token(model_name="gpt-4.1-mini", path=r"tests\kitten.png") == 856

    # remove the copied files
    os.remove(r"tests\kitten.jpeg")
    os.remove(r"tests\kitten.png")
