import math


def calculate_image_tokens(width, height, max_tokens, patch_size=32):
    # Step 1: Calculate number of patches without scaling
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


if __name__ == "__main__":
    print(calculate_image_tokens(1600, 1600, 1536))
