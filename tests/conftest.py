from pathlib import Path

# Constants
JPG_FILE_PATH = str(Path("tests") / "image_folder" / "kitten.jpg")
JPEG_FILE_PATH = str(Path("tests") / "image_folder" / "kitten.jpeg")
PNG_FILE_PATH = str(Path("tests") / "image_folder" / "kitten.png")
JPEG_URL = "https://drive.usercontent.google.com/download?id=1iNJTKFYyW09lVStpWqdSfKjw3ktEc215&export=download&authuser=1&confirm=t&uuid=acc70c89-fcb3-451a-9700-2e59e2a3e46a&at=AN8xHoqd_Asknqn8z-dpjPbB1HJO:1752041579483"
JPG_URL = "https://drive.usercontent.google.com/download?id=1cdTsrL3HDpXvADHHNxEOMtVP2wGNjszm&export=download&authuser=1&confirm=t&uuid=784dad7a-5ec6-4d2e-9604-2e22efb1f049&at=AN8xHorOo7QTvxOmxt4OBedxv49N:1752041386063"
PNG_URL = "https://drive.usercontent.google.com/download?id=1GQ0MG6XGeH7Rn0gGX1z47cYfk-tsLIJT&export=download&authuser=1&confirm=t&uuid=bb4292d1-43cd-4891-a39b-8ad225cbf105&at=AN8xHoptjBYOR2OMy5OQgJe24RRr:1752041516513"
CACHE_TEST_IMAGE_URL = "https://static.vecteezy.com/system/resources/thumbnails/002/098/203/small/silver-tabby-cat-sitting-on-green-background-free-photo.jpg"

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
