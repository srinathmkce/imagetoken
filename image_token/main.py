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
    check_case,
    is_image_url,
    get_image_url_field
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
from urllib.parse import unquote

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


def process_huggingFace_data(model_name: str, dataset: str, config: str = "default"):
    """
    Calculate and return the total number of tokens for a given hugging face Dataset.

    This function processes images present in hugging face dataset, calculates the number of tokens
    for each image based on the given model configuration, and optionally saves the results to
    a file. The total number of tokens includes the prefix tokens.

    Args:
        model_name (str): The name of the model to use for token calculation.
        dataset (str): The name of the hugging face dataset.
        config (str, optional): The configuration of the dataset. Defaults to "default".

    Returns:
        int: The total number of tokens calculated.
    """
    # Decode URL-encoded dataset name if necessary
    dataset = unquote(dataset)
    
    model_config = openai_config[model_name]
    max_token = model_config['max_tokens']
    typeofcase = check_case(dataset=dataset, config=config)

    if typeofcase == 'Case 1':
        print("Processing Case 1: Using API with statistical sampling")
        url = f'https://datasets-server.huggingface.co/statistics?dataset={dataset}&config={config}&split=train'
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            size = data.get("num_examples", 0)
            
            if size == 0:
                print("No examples found in dataset")
                return 0
            
            image_features = [f for f in data.get("statistics", []) if f.get("column_type") == "image"]
            
            if not image_features:
                print("No image columns found in dataset")
                return 0
            
            # Currently processing only first image column
            tokencount = 0
            
            # Calculate sample size based on standard deviation
            std_dev = image_features[0]["column_statistics"]["std"]
            sample_size = math.ceil(((1.96 * std_dev) / 50) ** 2)
            corrected_sample_size = math.ceil(sample_size / (1 + ((sample_size - 1) / size)))
            

            corrected_sample_size = min(corrected_sample_size, size)
            
            print(f"Dataset size: {size}")
            print(f"Calculated sample size: {sample_size}")
            print(f"Corrected sample size: {corrected_sample_size}")
            
            numbers = list(range(0, size))
            sample_rows = random.sample(numbers, corrected_sample_size)
            
            sample_token_sum = 0
            successful_samples = 0
            
            for i in sample_rows:
                try:
                    row_data_url = f'https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset={i}&length=1'
                    response = requests.get(row_data_url)
                    response.raise_for_status()
                    row_data = response.json()
                    
                    # Get image dimensions
                    height = row_data["rows"][0]['row']['image']['height']
                    width = row_data["rows"][0]['row']['image']['width']
                    
                    token = calculate_image_tokens(
                        model_name=model_name, 
                        width=width, 
                        height=height, 
                        max_tokens=max_token, 
                        model_config=model_config
                    )
                    
                    sample_token_sum += token
                    successful_samples += 1
                    print(f"Sample {successful_samples}: {width}x{height} -> {token} tokens")
                    
                except Exception as row_error:
                    print(f"Error processing row {i}: {row_error}")
                    continue
            
            if successful_samples > 0:
                # Calculate population estimate
                population_token_sum = (sample_token_sum * size) / successful_samples
                tokencount = population_token_sum
                
                print(f"\nResults:")
                print(f"Successful samples: {successful_samples}")
                print(f"Total tokens in sample: {sample_token_sum}")
                print(f"Estimated population tokens: {population_token_sum:.0f}")
                
                print("\nImage features found:")
                for feature in typeofcase[1]:
                    print(f"  - {feature}")
                
                return int(tokencount)
            else:
                print("No successful samples processed")
                return 0
                
        except Exception as e:
            print(f"Exception in Case 1: {e}")
            return 0

    elif typeofcase == 'Case 2':
        print("Processing Case 2: Using streaming dataset")
        url = f'https://datasets-server.huggingface.co/statistics?dataset={dataset}&config={config}&split=train'
        
        try:
            # Getting dataset size
            response = requests.get(url)
            response.raise_for_status()
            api_data = response.json()
            size = api_data.get("num_examples", 0)
            
            if size == 0:
                print("No examples found in dataset")
                return 0 
            
            # Use 10% sample or minimum 100 samples, whichever is larger
            sample_limit = max(int(size * 0.1), min(100, size))
            print(f"Dataset size: {size}")
            print(f"Sample limit: {sample_limit}")
            
            print(f"Loading dataset: {dataset}")
            dataset_stream = load_dataset(dataset, split='train', streaming=True)
            
            sample_token_sum = 0
            successful_samples = 0
            
            for i, example in enumerate(dataset_stream):
                try:
                    pil_image = example["image"]
                    width, height = pil_image.size
                    print(f"{i+1}. Image size: {width}x{height}")
                    
                    token = calculate_image_tokens(
                        model_name=model_name, 
                        width=width,
                        height=height, 
                        max_tokens=max_token, 
                        model_config=model_config
                    )
                    
                    sample_token_sum += token
                    successful_samples += 1
                    print(f"Tokens: {token}")
                    
                    if successful_samples >= sample_limit:
                        break
                        
                except Exception as img_error:
                    print(f"Error processing image {i+1}: {img_error}")
                    continue
            
            if successful_samples > 0:
                # Calculate population estimate
                avg_tokens_per_image = sample_token_sum / successful_samples
                population_token_sum = avg_tokens_per_image * size
                
                print(f"\nResults:")
                print(f"Successful samples: {successful_samples}")
                print(f"Total tokens in sample: {sample_token_sum}")
                print(f"Average tokens per image: {avg_tokens_per_image:.2f}")
                print(f"Estimated population tokens: {population_token_sum:.0f}")
                
                return int(population_token_sum)
            else:
                print("No successful samples processed")
                return 0
                
        except Exception as e:
            print(f"Exception in Case 2: {e}")
            return 0

    elif typeofcase == 'Case 3':
        url_stat = f'https://datasets-server.huggingface.co/statistics?dataset={dataset}&config={config}&split=train'
        response = requests.get(url_stat)
        data = response.json()
        size = data.get("num_examples", 0)
        if size == 0:
            print("No examples found in dataset")
            return 0

        # Step 2: Find image field name by checking first row
        url_first_row = f'https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset=0&length=1'
        response = requests.get(url_first_row)
        row_data = response.json()
        first_row = row_data["rows"][0]["row"]

        image_field = get_image_url_field(first_row)
        if not image_field:
            print("No image URL field detected in first row")
            return 0

        print(f"Detected image URL field: {image_field}")
        print(f"Dataset size: {size}")

        # Step 3: Sample 30% or some fixed size , whichever is smaller
        sample_size = min(math.ceil(size * 0.3), 15)
        sample_rows = random.sample(range(0, size), sample_size)

        print(f"Sampling {sample_size} rows for image URL extraction")

        image_urls = []
        for i in sample_rows:
            try:
                row_url = f'https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset={i}&length=1'
                response = requests.get(row_url)
                row = response.json()["rows"][0]["row"]
                image_url = row.get(image_field)

                if isinstance(image_url, dict):
                    # try nested values
                    for v in image_url.values():
                        if is_image_url(v):
                            image_url = v
                            break
                if is_image_url(image_url):
                    image_urls.append(image_url)
                    print(f"[{len(image_urls)}] {image_url}")

            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue

        if not image_urls:
            print("No valid image URLs found in sampled data")
            return 0
    
        num_tokens = get_token(model_name=model_name, path=image_urls)

        # Step 5: Estimate population tokens
        avg_tokens = num_tokens / len(image_urls)
        estimated_total = avg_tokens * size

        print(f"\nResults:")
        print(f"Sampled URLs: {len(image_urls)}")
        print(f"Total tokens for sample: {num_tokens}")
        print(f"Average tokens per image: {avg_tokens:.2f}")
        print(f"Estimated total tokens for dataset: {estimated_total:.0f}")

        return int(estimated_total)

    
    else:
        print(f"Unknown case type: {typeofcase[0]}")
        return 0