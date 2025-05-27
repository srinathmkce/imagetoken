# imagetoken

Utility to estimate the number of tokens in an image or directory of images for OpenAI models.

## Install using pip

```bash
pip install image-token
```

## Supported models

- gpt-4.1-mini
- gpt-4.1-nano

## Usage

To get the number of tokens for a single image
```python
from image_token import get_token
num_tokens = get_token(model_name="gpt-4.1-mini", path=r"kitten.jpg")
```

To get the number of tokens for a directory of images
```python
from image_token import get_token
num_tokens = get_token(model_name="gpt-4.1-mini", path=r"image_folder")
```

To get the estimated cost of generating text from an image or directory of images
```python
from image_token import get_cost
cost = get_cost(model_name="gpt-4.1-nano", system_prompt_tokens=300 * 100, approx_output_tokens=100 * 100, path=r"image_folder")
```

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