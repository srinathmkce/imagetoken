import json
from image_token.validate import (
    check_if_path_is_file,
    check_if_path_is_folder,
    check_allowed_extensions,
    check_valid_model,
    is_url,
    is_multiple_urls
)
from image_token.utils import (
    read_image_dims,
    calculate_cost,
    list_all_images,
    process_image_from_url,
    check_case
)
from image_token.core import calculate_image_tokens
from image_token.config import openai_config
from tqdm import tqdm
from pathlib import Path
from image_token.caching_utils import ImageDimensionCache
import requests
import math
import random
from PIL import Image
from datasets import load_dataset , Image

def process_image(path: str, model_config: dict, model_name: str = None):
    """
    Process an image and calculate the number of tokens in it.

    Args:
        path (str): The path to the image file.
        model_config (dict): The configuration for the model.

    Returns:
        int: The number of tokens in the image.
    """
    check_allowed_extensions(path=path)

    width, height = read_image_dims(path=path)

    max_tokens = model_config["max_tokens"]

    num_tokens = calculate_image_tokens(
        model_name=model_name,
        width=width,
        height=height,
        max_tokens=max_tokens,
        model_config=model_config,
    )

    return num_tokens


def get_cost(
    model_name: str,
    system_prompt_tokens: int,
    approx_output_tokens: int,
    path: Path | str,
    save_to: str = None,
    prefix_tokens: int = 9,
):
    """
    Calculate and return the estimated cost of generating text from an image or directory of images.

    Args:
        model_name (str): The name of the model to use.
        system_prompt_tokens (int): The number of tokens in the system prompt.
        approx_output_tokens (int): The approximate number of tokens in the output.
        path (str): The path to the image file or directory of images.
        save_to (str): The path to save the output to.
        prefix_tokens (int): The number of prefix tokens to use. Defaults to 9.

    Returns:
        float: The estimated cost in dollars.
    """

    model_config = openai_config[model_name]
    input_tokens = get_token(
        model_name=model_name, path=path, prefix_tokens=prefix_tokens, save_to=save_to
    )
    cost = calculate_cost(
        input_tokens=system_prompt_tokens + input_tokens,
        output_tokens=approx_output_tokens,
        config=model_config,
    )
    return cost


def get_token(model_name: str, path: str, prefix_tokens: int = 9 , save_to: str = None ):
    """
    Calculate and return the total number of tokens for a given image or directory of images.

    This function processes an image or a directory of images, calculates the number of tokens
    for each image based on the given model configuration, and optionally saves the results to
    a file. The total number of tokens includes the prefix tokens.

    Args:
        model_name (str): The name of the model to use for token calculation.
        path (str): The path to the image file or directory containing images.
        prefix_tokens (int, optional): The number of tokens to add as a prefix. Defaults to 9.
        save_to (str, optional): The path to a file where the results should be saved. If None,
                                 the results are not saved to a file.

    Returns:
        int: The total number of tokens calculated.
    """

    check_valid_model(model_name=model_name)
    model_config = openai_config[model_name]
    result_dict = {}
    total_tokens = 0

    if check_if_path_is_file(path=path):
        num_tokens = process_image(
            path=path, model_config=model_config, model_name=model_name
        )
        total_tokens = num_tokens + prefix_tokens
        result_dict[path] = total_tokens
    elif check_if_path_is_folder(path=path):
        image_files = list_all_images(path=path)
        total_tokens = 0
        for image_path in tqdm(image_files):
            num_tokens = process_image(
                path=image_path, model_config=model_config, model_name=model_name
            )
            total_tokens += num_tokens + prefix_tokens
            result_dict[image_path] = num_tokens + prefix_tokens
    elif is_url(path=path):
        with ImageDimensionCache() as cache:
            num_tokens = process_image_from_url(
            url=path,
            model_config=model_config,
            model_name=model_name,
            cache = cache
        )
        total_tokens = num_tokens + prefix_tokens
        result_dict[path] = total_tokens
    elif is_multiple_urls(urls = path):
        with ImageDimensionCache() as cache:
            for url in path : 
                num_tokens = process_image_from_url(
                    url = url,
                    model_name=model_name,
                    model_config=model_config,
                    cache=cache
                )
                total_tokens += num_tokens + prefix_tokens
                result_dict[url] = num_tokens + prefix_tokens
    else:
        raise ValueError(
            f"Invalid input path or URL: '{path}'. The given input is not a valid file, folder, or URL."
        )

    if save_to:
        with open(save_to, "w") as f:
            json.dump(result_dict, f, indent=4)

    return total_tokens

def process_huggingFace_data(model_name: str, dataset: str , config: str = "default"):
    """
    Calculate and return the total number of tokens for a given hugging face Dataset.

    This function processes an images present in hugging face dataset, calculates the number of tokens
    for each image based on the given model configuration, and optionally saves the results to
    a file. The total number of tokens includes the prefix tokens.

    Args:
        model_name (str): The name of the model to use for token calculation.
        path (str): The path to the hugging face url.
        prefix_tokens (int, optional): The number of tokens to add as a prefix. Defaults to 9.
        sampling (bool, optional): This specifies whether to estimate the tokens based on sampling or process full data 
        save_to (str, optional): The path to a file where the results should be saved. If None,
                                 the results are not saved to a file.

    Returns:
        int: The total number of tokens calculated.
    """
    model_config = openai_config[model_name]
    max_token = model_config['max_tokens']
    typeofcase = check_case(dataset = dataset , config = config)

    if typeofcase[0] == 'Case 1':
        print("print case 1")
        url = f'https://datasets-server.huggingface.co/statistics?dataset={dataset}&config={config}&split=train'
        try:
            response = requests.get(url)
            data = response.json()
            size = data.get("num_examples", '')
            image_features = [f for f in data.get("statistics") if f.get("column_type") == "image"]
            # currently processing only first image col
            tokencount = 0  # token count for diff column
            # for i in range(len(image_features)): # ... if want to process all the image column
            std_dev = image_features[0]["column_statistics"]["std"]
            sample_size = math.ceil(((1.96 * std_dev) / 50) ** 2)
            corrected_samplesize = math.ceil(sample_size / (1 + ((sample_size - 1) / size)))
            numbers = list(range(1, size))
            sample_rows = random.sample(numbers, corrected_samplesize)
            sample_token_sum = 0 # token count within each column for a col
            for i in sample_rows :
                row_data = f'https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset={i}&length={1}'
                response = requests.get(row_data)
                data = response.json()
                height = data["rows"][0]['row']['image']['height']
                widht = data["rows"][0]['row']['image']['width']
                token = calculate_image_tokens(model_name=model_name , width=widht , height=height , max_tokens=max_token , model_config=model_config)
                sample_token_sum = token + sample_token_sum
                print(token)
            population_token_sum = ( sample_token_sum * size ) / sample_size # for a col
            # tokencount = population_token_sum + tokencount

            # return tokencount

            # print(token_sum)
            print(population_token_sum)
            print(sample_size)
            print(corrected_samplesize)

            for feature in typeofcase[1]:
                print(feature)
        except Exception as e:
            print(f"Exception: {e}")

        # get the statistics for the dataset
        # extract the no of rows
        # decide the no of sample 
        # useing limit and offset get the row api call
        # get the url from that row then use url funtion to get the token
        # do for all the sample images and extrapolate to entire population
    elif typeofcase[0] == 'Case 3':
        print("print case 1")
        # data = data
        url = f'https://datasets-server.huggingface.co/statistics?dataset={dataset}&config={config}&split=train'
        try:
            response = requests.get(url)
            data = response.json()
            size = data.get("num_examples", '')
            sample_limit = size * 0.1
            data = dataset.load_datast(dataset , split = config , streaming = True)
            sample_token_sum = 0
            for i, example in enumerate(dataset):
                pil_image = example["image"]
                width, height = pil_image.size
                print(f"{i+1}. Image size: {width}x{height}")
                token = calculate_image_tokens(model_name=model_name , width=widht , height=height , max_tokens=max_token , model_config=model_config)
                sample_token_sum = token + sample_token_sum

                if i + 1 >= sample_limit:
                    break
            population_token_sum = ( sample_token_sum * size ) / sample_size # for a col

        except Exception as e:
            print(f"Exception: {e}")
        
        # width, height = image.size
        # get the statisitcs
        # decide how much to stream
        # then using pil get the image dimensions and get the token
        # keep adding token and extrapolite to entire population

    elif typeofcase[0] == 'Case 2':
        print("print case 1")
        # get the statistics
        # work on util to find the col which has the image url 
        # need to fetch the first row and use .png .jpg .jpeg endiing feature to get the col name
        # then use case 1 logic
    else :
        print("hello")