import base64
import pytest
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from image_token.langchain_callback import simulate_image_token_cost


def encode_image_to_data_url(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{image_base64}"


def build_messages_with_image(image_path: Path):
    image_data_url = encode_image_to_data_url(image_path)
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        ),
    ]


def build_messages_with_multiple_images(folder_path: Path, limit: int = 3):
    system = SystemMessage(content="You are a helpful assistant.")
    contents = []

    image_files = list(folder_path.glob("*.jpg"))[:limit]
    for img_path in image_files:
        contents.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(img_path)},
            }
        )

    return [system, HumanMessage(content=contents)]


def test_simulate_single_image():
    """Test that a single image is processed correctly and returns cost and tokens."""
    llm = ChatOpenAI(model="gpt-4.1-nano")
    image_path = str(Path("tests") / "image_folder" / "kitten.jpg")
    messages = build_messages_with_image(image_path)

    result = simulate_image_token_cost(llm, messages)

    assert isinstance(result, dict)
    assert result["tokens"] == pytest.approx(1320 , abs = 10)
    assert result["cost"] == 0.00013


def test_func_test_all_images():
    llm = ChatOpenAI(model="gpt-4.1-nano")
    image_folder = Path("tests/image_folder")

    image_files = (
        list(image_folder.glob("*.jpg"))
        + list(image_folder.glob("*.jpeg"))
        + list(image_folder.glob("*.png"))
    )

    messages = [
        SystemMessage(content="You are a helpful assistant."),
    ]
    for image_path in image_files:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{image_base64}"

        messages.append(
            HumanMessage(
                content=[{"type": "image_url", "image_url": {"url": image_data_url}}]
            )
        )
    result = simulate_image_token_cost(llm, messages)

    assert isinstance(result, dict)
    assert result["tokens"] == pytest.approx(5690 , abs = 30)
    assert result["cost"] == 0.00057


def test_when_message_is_empty():
    llm = ChatOpenAI(model="gpt-4.1-nano")
    image_path = str(Path("tests") / "image_folder" / "kitten.jpg")
    messages = build_messages_with_image(image_path)

    # Empty the message
    messages = []
    result = simulate_image_token_cost(llm, messages)
    assert isinstance(result, dict)
    assert result["tokens"] == 13
    assert result["cost"] == 0


def test_when_human_message_contains_string():
    llm = ChatOpenAI(model="gpt-4.1-nano")
    image_path = str(Path("tests") / "image_folder" / "kitten.jpg")
    messages = build_messages_with_image(image_path)
    # Add a string message
    messages.append(HumanMessage(content="Hello, world!"))
    messages = messages[-1]
    result = simulate_image_token_cost(llm, messages)
    assert isinstance(result, dict)
    assert result["tokens"] == 13
    assert result["cost"] == 0
