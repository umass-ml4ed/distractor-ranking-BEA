import openai
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError, APIConnectionError
import os
import json
import time
from tqdm import tqdm
from omegaconf import OmegaConf
import concurrent.futures

api_keys = [""]

def get_saved_cache(cache_filename: str):
    if os.path.exists(cache_filename):
        print(f"Loading {cache_filename}...")
        with open(cache_filename, encoding="utf-8") as cache_file:
            return json.load(cache_file)
    return {}

class OpenAIInterface:
    delay_time = 0.5
    decay_rate = 0.8
    cache = {} # Maps cache filename to prompt to result

    # Max number of prompts per request
    API_MAX_BATCH = 20
    # Codex max rate limit is 40k tokens per minute
    # https://platform.openai.com/docs/guides/rate-limits/error-mitigation
    MAX_TPM = float(40000)
    CHAT_GPT_MODEL_NAME = ["gpt-4-turbo-preview", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"]

    def __init__(self, openaicfg):
        self.openaicfg = openaicfg

    @staticmethod
    def get_cache(openaicfg):
        cache_filename = f"oai_cache_{openaicfg.model}.json"
        if cache_filename not in OpenAIInterface.cache:
            OpenAIInterface.cache[cache_filename] = get_saved_cache(cache_filename)
        return OpenAIInterface.cache[cache_filename]

    @staticmethod
    def save_cache():
        for cache_filename, cache in OpenAIInterface.cache.items():
            print(f"Saving {cache_filename}...")
            with open(cache_filename, "w") as cache_file:
                json.dump(cache, cache_file)

    @staticmethod
    def getCompletionForAllPrompts(openaicfg, prompts, batch_size=10, dynamic_retry=True, use_parallel=True):
        # 10 is a sweet spot for batch size, pro tip from Alex
        cache = OpenAIInterface.get_cache(openaicfg)
        uncached_prompts = [prompt for prompt in prompts if prompt not in cache]
        print(f"Found {len(prompts) - len(uncached_prompts)} prompts in cache!")
        batch_size = min(batch_size, OpenAIInterface.API_MAX_BATCH)
        prompt_batches = [uncached_prompts[i : i + batch_size] for i in range(0, len(uncached_prompts), batch_size)]
        for prompt_batch in tqdm(prompt_batches):
            if use_parallel:
                batch_responses = OpenAIInterface.getParallelCompletion(openaicfg, prompt_batch, dynamic_retry=dynamic_retry)
            else:
                batch_responses = OpenAIInterface.getBatchCompletion(openaicfg, prompt_batch, dynamic_retry=dynamic_retry)
            for prompt, response in zip(prompt_batch, batch_responses):
                cache[prompt] = response
        return [cache[prompt] for prompt in prompts]

    @staticmethod
    def getBatchCompletion(openaicfg, prompts, dynamic_retry=False):
        try:
            batch_response = OpenAIInterface.getCompletion(openaicfg, prompts, dynamic_retry)
            batch_completions = [None] * len(prompts)
            # match completions by index
            for choice in batch_response.choices:
                batch_completions[choice.index] = choice
            return batch_completions
        except Exception as e:
            print(e)
            raise e

    @staticmethod
    def getParallelCompletion(openaicfg, prompts, dynamic_retry=False):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                # Submit requests to threads
                futures = [
                    executor.submit(OpenAIInterface.getCompletion, openaicfg, [prompt], dynamic_retry)
                    for prompt in prompts
                ]

                # Wait for all to complete
                concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

                # Accumulate results
                results = [future.result().choices[0] for future in futures]
                return results
        except Exception as e:
            print(e)
            raise e

    @staticmethod
    def getCompletion(openaicfg, prompt, dynamic_retry):
        if dynamic_retry:
            # TODO this should be non-blocking
            time.sleep(OpenAIInterface.delay_time)

        # TODO handle multiple api keys
        openai.api_key = api_keys[0]
        try:
            if openaicfg.stop is None:
                stops = None
            else:
                stops = OmegaConf.to_container(openaicfg.stop) # Adapting the OmegaConf list to a python list
            if openaicfg.model in OpenAIInterface.CHAT_GPT_MODEL_NAME:
                assert len(prompt) == 1, "Chat only supports one prompt"
                response = openai.ChatCompletion.create(
                    model=openaicfg.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a skilled middle school math teacher with expertise in crafting high-quality incorrect answers(distractors) for math multiple-choice questions."
                        },
                        {
                            "role": "user",
                            "content": prompt[0]
                        }
                    ],
                    temperature=openaicfg.temperature,
                    max_tokens=openaicfg.max_tokens,
                    top_p=openaicfg.top_p,
                    frequency_penalty=openaicfg.frequency_penalty,
                    presence_penalty=openaicfg.presence_penalty,
                    request_timeout=45
                )
            else:
                response = openai.Completion.create(
                    model=openaicfg.model,
                    prompt=prompt,
                    temperature=openaicfg.temperature,
                    max_tokens=openaicfg.max_tokens,
                    top_p=openaicfg.top_p,
                    frequency_penalty=openaicfg.frequency_penalty,
                    presence_penalty=openaicfg.presence_penalty,
                    stop=stops,
                    logprobs=openaicfg.logprobs,
                    echo=openaicfg.echo
                )
            if dynamic_retry:
                OpenAIInterface.delay_time = max(OpenAIInterface.delay_time * OpenAIInterface.decay_rate, 0.1)
            return response
        except (RateLimitError, Timeout, APIError, ServiceUnavailableError, APIConnectionError) as exc:
            print(openai.api_key, exc)
            if dynamic_retry:
                OpenAIInterface.delay_time = min(OpenAIInterface.delay_time * 2, 30)
                print(f"Backoff request detected, increasing delay to {OpenAIInterface.delay_time} seconds")
                return OpenAIInterface.getCompletion(openaicfg, prompt, dynamic_retry)
            raise e
        except Exception as e:
            print(e)
            raise e
