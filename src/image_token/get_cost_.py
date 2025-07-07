from image_token.get_tokens import GetTokens

async def get_cost_async_input_tokens(
    model_name: str,
    path: str,
    prefix_tokens: int,
    save_to: str = None,
) -> int:
    """
    Helper to fetch token count asynchronously for cost calculation.
    """
    obj = GetTokens(model_name, path, prefix_tokens, save_to)
    return await obj.get_token_async()