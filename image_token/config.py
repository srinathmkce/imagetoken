openai_config = {
    "gpt-5-mini": {
        "factor": 1.2,
        "max_tokens": 1536,
        "input_tokens": 0.25,
        "output_tokens": 2.00,
    },
    "gpt-5-nano": {
        "factor": 1.5,
        "max_tokens": 1536,
        "input_tokens": 0.05,
        "output_tokens": 0.40,
    },
    "gpt-5": {
        "factor": 2.46,
        "max_tokens": 400000,
        "input_tokens": 1.25,
        "output_tokens": 10.00,
    },
    "gpt-5-chat-latest": {
        "factor": 2.46,
        "max_tokens": 400000,
        "input_tokens": 1.25,
        "output_tokens": 10.00,
    },
    "gpt-4.1-mini": {
        "factor": 1.62,
        "max_tokens": 1536,
        "input_tokens": 0.40,
        "output_tokens": 1.60,
    },
    "gpt-4.1-nano": {
        "factor": 2.46,
        "max_tokens": 1536,
        "input_tokens": 0.10,
        "output_tokens": 0.40,
    },
    "o4-mini": {
        "factor": 1.72,
        "max_tokens": 1536,
        "input_tokens": 1.10,
        "output_tokens": 4.40,
    },
    "gpt-4o-mini": {
        "factor": 1.72,
        "max_tokens": 128000,
        "input_tokens": 0.15,
        "output_tokens": 0.60,
    },
    "gpt-4o": {
        "factor": 1.72,
        "max_tokens": 128000,
        "input_tokens": 2.50,
        "output_tokens": 10.00,
    },
    "gpt-4.1": {
        "factor": 1.72,
        "max_tokens": 1000000,
        "input_tokens": 2.00,
        "output_tokens": 8.00,
    },
}


patch_models = {
    "gpt-5-mini": 1.62,
    "gpt-5-nano": 2.46,
    "gpt-4.1-mini": 1.62,
    "gpt-4.1-nano": 2.46,
    "o4-mini": 1.72,
}


tile_models = {
    "gpt-5": {"base": 70, "tile": 140},
    "gpt-5-chat-latest": {"base": 70, "tile": 140},
    "gpt-4.1": {"base": 85, "tile": 170},
    "gpt-4o": {"base": 85, "tile": 170},
    "gpt-4o-mini": {"base": 2833, "tile": 5667},
}

gemini_config = {
    "gemini-2.5-pro": {
        "pricing_tiers": [
            {
                "up_to_tokens": 200000,
                "input_cost_per_million_tokens": 1.25,
                "output_cost_per_million_tokens": 10.00,
            },
            {
                "up_to_tokens": float("inf"),
                "input_cost_per_million_tokens": 2.50,
                "output_cost_per_million_tokens": 15.00,
            },
        ]
    },
    "gemini-2.5-flash": {
        "pricing_tiers": [
            {
                "up_to_tokens": float("inf"),
                "input_cost_per_million_tokens": 0.10,
                "output_cost_per_million_tokens": 2.50,
            }
        ]
    },
    "gemini-2.5-flash-lite": {
        "pricing_tiers": [
            {
                "up_to_tokens": float("inf"),
                "input_cost_per_million_tokens": 0.10,
                "output_cost_per_million_tokens": 0.40,
            }
        ]
    },
    "gemini-2.5-flash-image-preview": {
        "pricing_tiers": [
            {
                "up_to_tokens": float("inf"),
                "input_cost_per_million_tokens": 0.30,
                "output_cost_per_million_tokens": 30.00,   # need some fixing
            }
        ]
    },
    "gemini-2.0-flash": {
        "pricing_tiers": [
            {
                "up_to_tokens": float("inf"),
                "input_cost_per_million_tokens": 0.10,
                "output_cost_per_million_tokens": 0.40,
            }
        ]
    },
    "gemini-2.0-flash-lite": {
        "pricing_tiers": [
            {
                "up_to_tokens": float("inf"),
                "input_cost_per_million_tokens": 0.075,
                "output_cost_per_million_tokens": 0.030,
            }
        ]
    },
}


# initally for gemini models keeping input form like text image video cause in future if want to expand to support them

#   "gemini-1.5-pro": {
        # "pricing_tiers": [
            # {
                # "up_to_tokens": 128000,
                # "input_cost_per_million_tokens": 1.25,
                # "output_cost_per_million_tokens": 5.00,
            # },
            # {
                # "up_to_tokens": float("inf"),
                # "input_cost_per_million_tokens": 2.50,
                # "output_cost_per_million_tokens": 10.00,
            # },
        # ]
    # },
    # "gemini-1.5-flash": {
        # "pricing_tiers": [
            # {
                # "up_to_tokens": 128000,
                # "input_cost_per_million_tokens": 0.075,
                # "output_cost_per_million_tokens": 0.30,
            # },
            # {
                # "up_to_tokens": float("inf"),
                # "input_cost_per_million_tokens": 0.15,
                # "output_cost_per_million_tokens": 0.60,
            # },
        # ]
    # },
    # "gemini-1.0-pro": {
        # "pricing_tiers": [
            # {
                # "up_to_tokens": float("inf"),
                # "input_cost_per_million_tokens": 0.50,
                # "output_cost_per_million_tokens": 1.50,
            # }
        # ]
    # },