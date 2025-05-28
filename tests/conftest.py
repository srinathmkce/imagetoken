import pytest
from pathlib import Path


@pytest.fixture
def expected_output_tokens(const_var: dict) -> dict:
    EXPECTED_OUTPUT_TOKENS = {
        const_var["GPT_4_1_MINI_MODEL_NAME"]: {
            const_var["JPG_FILE_PATH"]: 865,
            const_var["JPEG_FILE_PATH"]: 2473,
            const_var["PNG_FILE_PATH"]: 423,
            "total_tokens": 3761,
        },
        const_var["GPT_4_1_NANO_MODEL_NAME"]: {
            const_var["JPG_FILE_PATH"]: 1310,
            const_var["JPEG_FILE_PATH"]: 3750,
            const_var["PNG_FILE_PATH"]: 638,
            "total_tokens": 5698,
        },
    }
    return EXPECTED_OUTPUT_TOKENS

@pytest.fixture
def test_cases():
    return {
        (64, 64): 15,
        (128, 256): 60,
        (256, 128): 60,
        (300, 500): 268,
        (800, 200): 292,
        (512, 512): 423,
        (1024, 1024): 1667,
    }

@pytest.fixture
def const_var():
    return {
        "JPG_FILE_PATH": str(Path("tests") / "image_folder" / "kitten.jpg"),
        "JPEG_FILE_PATH": str(Path("tests") / "image_folder" / "kitten.jpeg"),
        "PNG_FILE_PATH": str(Path("tests") / "image_folder" / "kitten.png"),
        "GPT_4_1_MINI_MODEL_NAME": "gpt-4.1-mini",
        "GPT_4_1_NANO_MODEL_NAME": "gpt-4.1-nano",
    }


