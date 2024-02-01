import functools
import inspect
import os
import diskcache
import json


def cache_res(func):
    pwd = os.getcwd()
    cache = diskcache.Cache(pwd)
    """Cache a function that caches an input list of strings"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        key = (
            f"{func.__name__}-{functools._make_key([args[1][0]], kwargs, typed=False)}"
        )
        if (cached := cache.get(key)) is not None:
            print("Read value from cache")
            # Deserialize from JSON based on the return type
            return json.loads(cached)

        try:
            result = await func(*args, **kwargs)
            # Call the function and cache its result

            serialized_result = result
            cache.set(key, json.dumps(serialized_result))
            return result
        except Exception as e:
            # Handle the exception here
            print(f"An error occurred: {e}")
            return None

    return wrapper
