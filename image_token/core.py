import math


def calculate_image_tokens(width, height, patch_size=32, max_tokens=1536):
    # Step 1: Calculate number of patches without scaling
    """
    Calculates the number of image tokens based on the dimensions of the image.

    This function determines the number of patches that can be extracted from an
    image with given width and height. It first calculates the total number of
    patches without any scaling, and if the total number of patches exceeds the
    maximum allowed tokens, it scales down the image dimensions to fit within
    the maximum token limit.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        patch_size (int, optional): The size of each patch. Defaults to 32.
        max_tokens (int, optional): The maximum number of tokens allowed.
                                    Defaults to 1536.

    Returns:
        int: The calculated number of image tokens.
    """

    patches_w = (width + patch_size - 1) // patch_size
    patches_h = (height + patch_size - 1) // patch_size
    total_patches = patches_w * patches_h

    if total_patches <= max_tokens:
        return total_patches

    # Step 2: Scaling required
    # Calculate shrink factor using given formula
    shrink_factor = math.sqrt((max_tokens * patch_size**2) / (width * height))
    scaled_width = width * shrink_factor
    scaled_height = height * shrink_factor

    # Calculate patch count from scaled dimensions
    patches_w_scaled = scaled_width / patch_size
    patches_h_scaled = scaled_height / patch_size

    # Fit the image to an integer number of patches (round down width)
    adjusted_patches_w = int(patches_w_scaled)
    adjusted_patches_h = int(patches_h_scaled)
    scale_adjust_w = adjusted_patches_w / patches_w_scaled
    scale_adjust_h = adjusted_patches_h / patches_h_scaled

    final_width = scaled_width * scale_adjust_w
    final_height = scaled_height * scale_adjust_h

    final_patches_w = int(final_width / patch_size)
    final_patches_h = int(final_height / patch_size)

    # Final token count is number of patches
    return final_patches_w * final_patches_h
