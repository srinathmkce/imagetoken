# imagetoken

Utility to estimate the number of tokens in an image or directory of images for OpenAI models.

## Install using pip

```bash
pip install image-token
```

## Supported models

- gpt-5
- gpt-5-mini
- gpt-5-nano
- gpt-4.1-mini
- gpt-4.1-nano
- gpt-4.1
- o4-mini
- gpt-4o
- gpt-4o-mini

> **Note**  
> The token count provided by `image-token` is an **estimation**.  
> Actual tokenization may vary by **±1–5 tokens**, depending on the model implementation.

## Usage

To get the number of tokens for a single image
```python
from image_token import get_token
num_tokens = get_token(model_name="gpt-5-mini", path=r"kitten.jpg")
```

To get the number of tokens for a directory of images
```python
from image_token import get_token
num_tokens = get_token(model_name="gpt-5-mini", path=r"image_folder")
```

To get the number of tokens for a URL
```python
from image_token import get_token
num_tokens = get_token(model_name="gpt-5-mini", path=r"https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.jpeg")
```

To get the number of token for multiple URLS
```python
for image_token import get_token
urls = ["https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.jpeg"
,"https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.jpg"
,"https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.png"
]
num_tokens = get_token(model_name="gpt-5-mini",path=urls)
```

To get the estimated cost of generating text from an image or directory of images
```python
from image_token import get_cost
cost = get_cost(model_name="gpt-5-nano", system_prompt_tokens=300 * 100, approx_output_tokens=100 * 100, path=r"image_folder")
```

## Langchain integration

You can simulate the langchain OpenAI call and calculate the input token and cost
```python
import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from image_token import simulate_image_token_cost


llm = ChatOpenAI(model="gpt-5-nano")

path = str(Path("tests") / "image_folder" / "kitten.jpg")

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

result = simulate_image_token_cost(llm, messages)

print(result)

```

You can simulate the langchain OpenAI call and calculate the input token and cost using URL
```python

import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from image_token import simulate_image_token_cost

llm = ChatOpenAI(model="gpt-5-nano")

image_data_url = "https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.jpeg"

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    ),
]

result = simulate_image_token_cost(llm, messages)

print(result)


```

**Note:** `simulate_image_token_cost` **mocks** the **LangChain OpenAI API** and returns the result **without making an actual request** to the LangChain endpoint. You can **simulate the call before executing** `llm.invoke(messages)`. The **token count and cost** calculated are based **only on the input tokens**. To get the **accurate cost**, make sure to **include the output tokens** as well.


## Run unit tests

Perform test using `poetry` package manager

Add the package using the command

```bash
poetry add <package_name>
```
Install required packages using

```bash
poetry install
```

Create an `.env` file in the project root directory and set the openai key `OPENAI_API_KEY`

Run pytests with the following command

```bash
poetry run pytest -sv tests
```