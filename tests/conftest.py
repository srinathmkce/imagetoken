from pathlib import Path

# Constants
JPG_FILE_PATH = str(Path("tests") / "image_folder" / "kitten.jpg")
JPEG_FILE_PATH = str(Path("tests") / "image_folder" / "kitten.jpeg")
PNG_FILE_PATH = str(Path("tests") / "image_folder" / "kitten.png")
JPEG_URL = "https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.jpeg"
JPG_URL = "https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.jpg"
PNG_URL = "https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.png"
CACHE_TEST_IMAGE_URL = "https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.png"

GPT_5_MINI_MODEL_NAME = "gpt-5-mini"
GPT_5_NANO_MODEL_NAME = "gpt-5-nano"
GPT_5_MODEL_NAME = "gpt-5"
GPT_5_CHAT_LATEST_MODEL_NAME = "gpt-5-chat-latest"
GPT_4_1_MINI_MODEL_NAME = "gpt-4.1-mini"
GPT_4_1_NANO_MODEL_NAME = "gpt-4.1-nano"
O_4_MINI_MODEL_NAME = "o4-mini"
GPT_4_O_MINI_MODEL_NAME = "gpt-4o-mini"
GPT_4_O_MODEL_NAME = "gpt-4o"
GPT_4_1_MODEL_NAME = "gpt-4.1"

# MODEL_NAMES = [
#     GPT_4_1_MINI_MODEL_NAME, GPT_4_1_NANO_MODEL_NAME, O_4_MINI_MODEL_NAME,
#     GPT_4_O_MINI_MODEL_NAME, GPT_4_O_MODEL_NAME, GPT_4_1_MODEL_NAME,
# ]

MODEL_NAMES = [
    GPT_4_O_MINI_MODEL_NAME,
]


EXPECTED_OUTPUT_TOKENS = {
    GPT_5_MINI_MODEL_NAME: {
        JPG_URL: 648 - 6,           # 642
        JPEG_URL: 1839 - 6,         # 1833
        PNG_URL: 321 - 6,           # 315
        "total_tokens": (642 + 1833 + 315),  # 2790
    },
    GPT_5_NANO_MODEL_NAME: {
        JPG_URL: 807 - 6,           # 801
        JPEG_URL: 2295 - 6,         # 2289
        PNG_URL: 397 - 6,           # 391
        "total_tokens": (801 + 2289 + 391),  # 3481
    },
    GPT_5_MODEL_NAME: {
        JPG_URL: 639 - 6,           # 633
        JPEG_URL: 639 - 6,          # 633
        PNG_URL: 222 - 6,           # 216
        "total_tokens": (633 + 633 + 216),   # 1482
    },
    GPT_5_CHAT_LATEST_MODEL_NAME: {
        JPG_URL: 643 - 6,           # 637
        JPEG_URL: 643 - 6,          # 637
        PNG_URL: 223 - 6,           # 217
        "total_tokens": (637 + 637 + 217),   # 1491
    },
    GPT_4_1_MINI_MODEL_NAME: {
        JPG_FILE_PATH: 865,
        JPEG_FILE_PATH: 2473,
        PNG_FILE_PATH: 423,
        "total_tokens": 3761,
    },
    GPT_4_1_NANO_MODEL_NAME: {
        JPG_FILE_PATH: 1310,
        JPEG_FILE_PATH: 3750,
        PNG_FILE_PATH: 638,
        "total_tokens": 5698,
    },
    O_4_MINI_MODEL_NAME: {
        JPG_FILE_PATH: 918,
        JPEG_FILE_PATH: 2625,
        PNG_FILE_PATH: 449,
        "total_tokens": 3992,
    },
    GPT_4_O_MINI_MODEL_NAME: {
        JPG_FILE_PATH: 25510,
        JPEG_FILE_PATH: 25510,
        PNG_FILE_PATH: 25510,
        "total_tokens": 76530,
    },
    GPT_4_O_MODEL_NAME: {
        JPG_FILE_PATH: 774,
        JPEG_FILE_PATH: 774,
        PNG_FILE_PATH: 774,
        "total_tokens": 2322,
    },
    GPT_4_1_MODEL_NAME: {
        JPG_FILE_PATH: 774,
        JPEG_FILE_PATH: 774,
        PNG_FILE_PATH: 774,
        "total_tokens": 2322,
    },
}

test_cases = {
    (64, 64): 15,
    (128, 256): 60,
    (256, 128): 60,
    (300, 500): 268,
    (800, 200): 292,
    (512, 512): 423,
    (1024, 1024): 1667,
}

# Dimensions to test
test_dimensions = [
    (64, 64),
    (128, 256),
    (256, 128),
    (300, 500),
    (800, 200),
    (512, 512),
    (1024, 1024),
]

# Test data - 4 input types

test_inputs = {
    "file" : "tests/image_folder/kitten.jpg",
    "folder" : "tests/image_folder",
    "url" : JPEG_URL,
    "urls" : [JPEG_URL , JPG_URL , PNG_URL],
    "random" : "random text"
}

# Provide access to test functions
# @pytest.fixture(scope="session")
# def model_names():
#     return MODEL_NAME

# @pytest.fixture(scope="session")
# def expected_output_tokens():
#     return EXPECTED_OUTPUT_TOKENS

# @pytest.fixture(scope="session")
# def image_paths():
#     return {
#         "jpg": JPG_FILE_PATH,
#         "jpeg": JPEG_FILE_PATH,
#         "png": PNG_FILE_PATH,
#     }

# # Dynamic parametrize
# def test_cases():
#     return {
#             (64, 64): 15, (128, 256): 60, (256, 128): 60,
#             (300, 500): 268, (800, 200): 292, (512, 512): 423,
#             (1024, 1024): 1667,
#         }
