import pytest
import asyncio
from image_token import get_cost
from conftest import JPG_FILE_PATH, GPT_4_1_MINI_MODEL_NAME


@pytest.mark.asyncio
async def test_get_cost_async():
    """
    Test the async version of get_cost to ensure cost is correctly computed.
    """
    model_name = GPT_4_1_MINI_MODEL_NAME
    system_prompt_tokens = 300
    approx_output_tokens = 150
    path = JPG_FILE_PATH

    cost = await asyncio.to_thread(
        get_cost,
        model_name,
        system_prompt_tokens,
        approx_output_tokens,
        path,
        None,
        9,
        "async"
    )

    assert isinstance(cost, float), "Cost should be a float value"
    assert cost > 0, f"Cost should be positive, got {cost:.8f}"
    print(f"Estimated async cost: ${cost:.8f}")