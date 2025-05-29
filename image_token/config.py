openai_config = {
    "gpt-4.1-mini": {
        "factor": 1.62,
        "max_tokens": 1536,
        "input_tokens": 0.4,
        "output_tokens": 1.6,
    },
    "gpt-4.1-nano": {
        "factor": 2.46,
        "max_tokens": 1536,
        "input_tokens": 0.1,
        "output_tokens": 0.4,
    },
    "o4-mini": {
        "factor": 1.72,
        "max_tokens": 1536,
        "input_tokens": 1.1,
        "output_tokens": 4.4,
    },
    "gpt-4o-mini": {
        "factor": 1.72,
        "max_tokens": 128000,
        "input_tokens": 0.15,
        "output_tokens": 0.60
    },
    "gpt-4o": {
        "factor": 1.72,
        "max_tokens": 128000,
        "input_tokens": 2.5,
        "output_tokens": 10.0,
    },
    "gpt-4.1": {
        "factor": 1.72,
        "max_tokens": 1000000,
        "input_tokens": 2.5,
        "output_tokens": 10.0,
    },
}

# Models using patch-based method
patch_models = {
    "gpt-4.1-mini": 1.62,
    "gpt-4.1-nano": 2.46,
    "o4-mini": 1.72,
}

# Models using tile-based method
tile_models = {
    "gpt-4.1": {"base": 85, "tile": 170},
    "gpt-4o": {"base": 85, "tile": 170},
    "gpt-4o-mini": {"base": 2833, "tile": 5667},
}
