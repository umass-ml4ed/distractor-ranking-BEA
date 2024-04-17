from OpenAIInterface import OpenAIInterface as oAI
from omegaconf import OmegaConf
import time
import json
import re
    
def generate_completion(prompts, conf):
    print("Calling OpenAI API...")
    prompt_tic = time.time()
    if conf.model in oAI.CHAT_GPT_MODEL_NAME:
        prompt_responses = oAI.getCompletionForAllPrompts(conf, prompts, batch_size=20, use_parallel=True)
    else:
        prompt_responses = oAI.getCompletionForAllPrompts(conf, prompts, batch_size=10, use_parallel=False)
    prompt_toc = time.time()
    print("Called OpenAI API in", prompt_toc - prompt_tic, "seconds.")
    oAI.save_cache()
    predictions = []
    for prompt_response in prompt_responses:
        pred = prompt_response["text"] if "davinci" in conf.model else prompt_response["message"]["content"]
        predictions.append(pred)
    return predictions

    
def main():
    data_address = "./prompts/zero_shot_10.json"
    # data_address = "./prompts/zero_shot_3.json"
    # data_address = "./prompts/zero_shot_10_complement.json"
    # data_address = "./prompts/zero_shot_3_complement.json"
    
    with open (data_address) as f:
        prompts = json.load(f)
    
    print(len(prompts))
    print(prompts[0])
    # prompts = prompts[:50]

    oaicfg = {
        'model': "gpt-4-turbo-preview", #   gpt-4-turbo-preview
        'temperature' : 0.0,
        'max_tokens' : 1000, # 1000 350
        'top_p' : 1.0,
        'frequency_penalty' : 0.0,
        'presence_penalty' : 0.0,
        'stop' : [],
        'logprobs': None,
        'echo' : False
        }
    conf = OmegaConf.create(oaicfg)        
    
    predictions = generate_completion(prompts, conf)
    
    # for predction in predictions:
    #     print(predction)
    #     print("_____________________")
    
    result_address = "./predictions/zero_shot_10_predictions.json"
    # result_address = "./predictions/zero_shot_3_predictions.json"
    # result_address = "./predictions/zero_shot_10_complement_predictions.json"
    # result_address = "./predictions/zero_shot_3_complement_predictions.json"
    
    with open(result_address, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    

if __name__ == "__main__":
    main()
