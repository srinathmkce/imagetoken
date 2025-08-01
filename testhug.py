# import numpy as np
# from PIL import Image
# import os
# import shutil
# import math
# from datasets import load_dataset # For loading images from Hugging Face
# from image_token import get_token

# # --- Your Provided Image Token Calculation Logic ---
# # These functions are copied directly from your provided code.
# # No changes are made to their internal logic.


# # --- Dataset Information (Your Provided Bins) ---
# all_datasets = [
#     {
#         "name": "Train_Dataset",
#         "hist": [38, 226, 319, 97, 12, 21, 0, 82, 0, 5],
#         "bin_edges": [204, 487, 770, 1053, 1336, 1619, 1902, 2185, 2468, 2751, 3024]
#     },
#     {
#         "name": "Test_Dataset",
#         "hist": [8, 25, 45, 5, 0, 5, 1, 10, 0, 1],
#         "bin_edges": [228, 508, 788, 1068, 1348, 1628, 1908, 2188, 2468, 2748, 3024]
#     },
#     {
#         "name": "Validation_Dataset",
#         "hist": [29, 1, 40, 15, 1, 0, 0, 3, 0, 11],
#         "bin_edges": [418, 614, 810, 1006, 1202, 1398, 1594, 1790, 1986, 2182, 2376]
#     }
# ]

# # --- Configuration ---
# NUM_SAMPLES_FOR_AR_LEARNING = 100 # How many images to sample from HF dataset for AR analysis
# HF_DATASET_NAME = "naver-clova-ix/cord-v2" # The actual HF dataset to sample from (must have an 'image' feature)
# HF_DATASET_SPLIT = "train" # The split to sample from
# output_directory = "simulated_images_with_learned_aspect_ratios"

# # --- Helper Functions ---
# def get_representative_width(bin_index, bin_edges):
#     """Calculates a representative width for a bin (midpoint or last edge)."""
#     if bin_index < len(bin_edges) - 1:
#         return (bin_edges[bin_index] + bin_edges[bin_index + 1]) // 2
#     else:
#         return bin_edges[bin_index]

# def find_bin_index(image_width, bin_edges):
#     """Determines which bin an image width falls into for a given set of bin_edges."""
#     for i in range(len(bin_edges) - 1):
#         if bin_edges[i] <= image_width < bin_edges[i+1]:
#             return i
#     if image_width >= bin_edges[-1]: # For images equal to or larger than the last bin edge
#         return len(bin_edges) - 1
#     return -1 # Should ideally not be reached if bin_edges cover all possible widths

# # --- PHASE 1: Load and Analyze Sample Images to Learn Aspect Ratios Per Bin ---

# print(f"\n--- Loading first {NUM_SAMPLES_FOR_AR_LEARNING} images from '{HF_DATASET_NAME}' for aspect ratio analysis ---")
# sample_image_dimensions = []
# try:
#     # Load in streaming mode for large datasets, take first N examples
#     hf_dataset_stream = load_dataset(HF_DATASET_NAME, split=HF_DATASET_SPLIT, streaming=True)
    
#     # Iterate through the stream and collect dimensions of the first N images
#     for i, example in enumerate(hf_dataset_stream):
#         if i >= NUM_SAMPLES_FOR_AR_LEARNING:
#             break
#         if 'image' in example and isinstance(example['image'], Image.Image):
#             sample_image_dimensions.append(example['image'].size) # (width, height)
#         else:
#             print(f"  Warning: Example {i} from {HF_DATASET_NAME} does not contain an 'image' feature or it's not a PIL Image. Skipping.")
    
#     print(f"  Successfully collected dimensions for {len(sample_image_dimensions)} images.")
#     if not sample_image_dimensions:
#         print("  WARNING: No image dimensions collected. This may lead to less accurate estimations. Using default 1:1 aspect ratio.")

# except Exception as e:
#     print(f"  ERROR: Could not load or process images from Hugging Face dataset '{HF_DATASET_NAME}': {e}")
#     print("  Proceeding with default 1:1 aspect ratio for all images as a fallback.")
#     sample_image_dimensions = [] # Ensure it's empty if error occurred


# # Store aspect ratios per bin for each dataset type
# bin_aspect_ratios_per_dataset = {}

# for dataset_info in all_datasets:
#     dataset_name = dataset_info["name"]
#     bin_edges_for_this_dataset = dataset_info["bin_edges"]
    
#     # Initialize a list to hold ARs for each bin
#     aspect_ratios_for_bins = {i: [] for i in range(len(bin_edges_for_this_dataset))}
    
#     # Populate these lists with ARs from the sample images
#     for img_width, img_height in sample_image_dimensions:
#         current_aspect_ratio = img_width / img_height if img_height > 0 else 1.0
#         bin_idx = find_bin_index(img_width, bin_edges_for_this_dataset)
        
#         if bin_idx != -1 and bin_idx < len(bin_edges_for_this_dataset):
#             aspect_ratios_for_bins[bin_idx].append(current_aspect_ratio)
    
#     # Calculate average AR for each bin and store it
#     bin_aspect_ratios_per_dataset[dataset_name] = {}
#     for bin_idx, ar_list in aspect_ratios_for_bins.items():
#         if ar_list:
#             bin_aspect_ratios_per_dataset[dataset_name][bin_idx] = sum(ar_list) / len(ar_list)
#         else:
#             # Fallback: if no samples fell into this bin, use 1.0 (square)
#             bin_aspect_ratios_per_dataset[dataset_name][bin_idx] = 1.0 

# print("\n--- Learned Average Aspect Ratios Per Bin (from sampled images) ---")
# for ds_name, bin_ars in bin_aspect_ratios_per_dataset.items():
#     print(f"Dataset: {ds_name}")
#     for bin_idx, avg_ar in bin_ars.items():
#         # Display the range of the bin for context
#         bin_start = bin_edges_for_this_dataset[bin_idx]
#         bin_end = bin_edges_for_this_dataset[bin_idx+1] if bin_idx < len(bin_edges_for_this_dataset)-1 else 'End'
#         print(f"  Bin {bin_idx} (Width {bin_start} to {bin_end}): Avg AR = {avg_ar:.3f}")
# print("------------------------------------------------------------------")


# # --- Prepare Output Directory ---
# if os.path.exists(output_directory):
#     shutil.rmtree(output_directory)
#     print(f"Cleaned up existing directory: '{output_directory}'")
# os.makedirs(output_directory, exist_ok=True)
# print(f"All generated representative images will be saved in: '{output_directory}'")

# # --- Initialize running total for the grand estimate across ALL datasets ---
# total_estimated_tokens_across_all_datasets = 0

# # --- PHASE 2: Process Each Dataset (Train, Test, Validation) for Token Estimation ---
# for dataset_info in all_datasets:
#     current_dataset_name = dataset_info["name"]
#     hist_data = dataset_info["hist"]
#     bin_edges = dataset_info["bin_edges"]
#     model_name_for_ai = "gpt-4o-mini" # Your AI model name for token calculation

#     print(f"\n\n======== Processing Dataset: {current_dataset_name} ========")
#     current_dataset_total_tokens = 0 # Accumulator for this specific dataset

#     # Retrieve the learned aspect ratios for the current dataset's binning
#     # Ensure a fallback to a default (e.g., all 1.0s) if no ARs were learned for this dataset
#     learned_aspect_ratios = bin_aspect_ratios_per_dataset.get(current_dataset_name, {})
#     if not learned_aspect_ratios: # If the dataset name didn't exist in the learned ARs
#         print(f"  WARNING: No aspect ratios learned for {current_dataset_name}. Using default 1:1 for all its bins.")
#         learned_aspect_ratios = {i: 1.0 for i in range(len(bin_edges))} # Fallback to all square

#     # Iterate through each bin in the current dataset
#     for i in range(len(hist_data)):
#         hist_count = hist_data[i]
        
#         # Get the representative width for this bin (midpoint)
#         representative_width = get_representative_width(i, bin_edges)
        
#         # Use the learned aspect ratio for this specific bin (fallback to 1.0 if not found)
#         representative_aspect_ratio = learned_aspect_ratios.get(i, 1.0) 
        
#         # Calculate height based on representative width and learned aspect ratio
#         representative_height = round(representative_width / representative_aspect_ratio)
        
#         # Ensure dimensions are at least 1x1 to avoid PIL errors
#         representative_width = max(1, representative_width)
#         representative_height = max(1, representative_height)


#         # Generate a unique filename for the image
#         image_filename = os.path.join(output_directory, f"{current_dataset_name}_bin_{i+1}_w{representative_width}_h{representative_height}.png")

#         print(f"\n--- Processing Bin {i+1} for {current_dataset_name} ---")
#         print(f"  Representative Image Dimensions: Width={representative_width}, Height={representative_height} (Learned AR={representative_aspect_ratio:.2f})")
#         print(f"  Number of images in this bin (from hist_data): {hist_count}")

#         if hist_count == 0:
#             print(f"  Skipping: hist_count is 0 for this bin. No images to estimate.")
#             continue

#         try:
#             # 1. Create a Random Image (representing one image from this bin)
#             random_pixels = np.random.randint(0, 256, (representative_height, representative_width, 3), dtype=np.uint8)
#             random_image = Image.fromarray(random_pixels)

#             # 2. Save the Representative Image
#             random_image.save(image_filename)
#             print(f"  Successfully created and saved '{image_filename}'.")

#             # 3. Get Token Count for this single representative image using YOUR logic
#             num_tokens_for_single_image = get_token(model_name=model_name_for_ai, path=image_filename)
#             print(f"  Tokens for ONE image of this size (using your provided logic): {num_tokens_for_single_image}")

#             # 4. Calculate Estimated Tokens for ALL images in this bin
#             estimated_tokens_for_this_bin = num_tokens_for_single_image * hist_count
#             print(f"  Estimated total tokens for this bin ({hist_count} images): {estimated_tokens_for_this_bin}")

#             current_dataset_total_tokens += estimated_tokens_for_this_bin

#         except Exception as e:
#             print(f"  An error occurred while processing bin {i+1} (width {representative_width}, dataset {current_dataset_name}): {e}")

#     print(f"\n--- Summary for {current_dataset_name} ---")
#     print(f"Total estimated tokens for {current_dataset_name}: {current_dataset_total_tokens}")
#     total_estimated_tokens_across_all_datasets += current_dataset_total_tokens

# print(f"\n\n=======================================================")
# print(f"GRAND TOTAL estimated tokens across ALL datasets (Train + Test + Validation): {total_estimated_tokens_across_all_datasets}")
# print(f"All generated representative images can be found in the '{output_directory}' folder.")

# # Optional: To clean up all generated images and the directory after running, uncomment this:
# # if os.path.exists(output_directory):
# #     shutil.rmtree(output_directory)
# #     print(f"\nCleaned up all generated images and the '{output_directory}' directory.")

# from datasets import load_dataset


# dataset = load_dataset('naver-clova-ix/cord-v2', split='train')
# print(dataset.info)

# # count = gettokenfrom huggingface(dataset = " " , config = '' else default )
# import requests

# def process_huggingFace_data(model_name: str, dataset: str, config: str = "default"):

#     url = f'https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset=0&length=1'
#     print(f"Requesting URL: {url}")

#     try:
#         response = requests.get(url)
#         data = response.json()

#         # Case 2: API error due to dataset size
#         if "error" in data:
#             print("Case 2 ❌ - Dataset too large or server error")
#             print(f"Error: {data['error']}")
#             return

#         # Extract features from response
#         features = data.get("features", [])

#         # Detect Case 1: Contains actual Image type
#         has_image_type = any(f.get("type", {}).get("_type") == "Image" for f in features)

#         # Detect Case 3: URL-based image in string columns
#         has_url_like_feature = any("url" in f.get("name", "").lower() and f.get("type", {}).get("_type") == "Value" for f in features)

#         # Case 1: Image column explicitly defined
#         if has_image_type:
#             print("Case 1 ✅ - Image defined in metadata")
#             print(has_image_type)

#         # Case 3: URL image reference without Image type
#         elif has_url_like_feature:
#             print("Case 3 🌐 - Image accessible via URL field")

#         # Fallback: No image indication
#         else:
#             print("Unknown case ❓ - No image metadata or URL")

#         # Show sample row for inspection
#         print("Response data:")
#         print(data.get("rows", [{}])[0])

#     except Exception as e:
#         print("Case 3 ⚠️ - Request failed or unexpected response")
#         print(f"Exception: {e}")

# # Example usage
# process_huggingFace_data(model_name="hello", dataset="multimodal-reasoning-lab%2FZebra-CoT" , config="2D+Visual+Reasoning+-+Visual+Jigsaw")


# # https://datasets-server.huggingface.co/rows?dataset=multimodal-reasoning-lab%2FZebra-CoT&config=2D+Visual+Reasoning+-+Visual+Jigsaw&split=train&offset=0&length=1
# # https://datasets-server.huggingface.co/first-rows?dataset=multimodal-reasoning-lab%2FZebra-CoT&config=2D+Visual+Reasoning+-+Visual+Jigsaw&split=train

from image_token.main import process_huggingFace_data

process_huggingFace_data(model_name="gpt-4.1-nano" , dataset="allenai%2FCoSyn-400K" , config="chart")