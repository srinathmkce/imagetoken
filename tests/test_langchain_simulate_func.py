import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from image_token import simulate_image_token_cost


def test_langchain_single_image_func():
    llm = ChatOpenAI(model="gpt-4.1-nano")

    path = str(Path("tests") / "image_folder" / "kitten.jpg")

    # response = llm.invoke("What is the capital of France?", config={"callbacks": callbacks})
    with open(path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        ),
    ]

    # Simulate the image token cost
    simulated_result = simulate_image_token_cost(llm, messages)

    result = llm.invoke(messages)
    actual_input_tokens = result.usage_metadata["input_tokens"]

    assert actual_input_tokens == simulated_result["tokens"]


# def test_langchain_multiple_images_func():
#     llm = ChatOpenAI(model="gpt-4.1-nano", verbose=True)
#     image_folder = Path("tests/image_folder")

#     image_files = (
#         list(image_folder.glob("*.jpg"))
#         # + list(image_folder.glob("*.jpeg"))
#         + list(image_folder.glob("*.png"))
#     )

#     messages = [
#         SystemMessage(content="You are a helpful assistant."),
#     ]
#     for image_path in image_files:
#         with open(image_path, "rb") as image_file:
#             image_bytes = image_file.read()
#             image_base64 = base64.b64encode(image_bytes).decode("utf-8")
#             image_data_url = f"data:image/jpeg;base64,{image_base64}"

#         messages.append(
#             HumanMessage(
#                 content=[{"type": "image_url", "image_url": {"url": image_data_url}}]
#             )
#         )

#     # Simulate the image token cost
#     simulated_result = simulate_image_token_cost(llm, messages)

#     result = llm.invoke(messages)
#     actual_input_tokens = result.usage_metadata['input_tokens']

#     assert actual_input_tokens == simulated_result['tokens']
