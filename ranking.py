from typing import List
import argparse
import re
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from utils import initialize_seeds, device, str_to_dict_eedi_df, get_model
import csv

def get_data(args):
    if args.rank_10:
        train_data = str_to_dict_eedi_df(pd.read_csv("data/zero_shot_more_than_10_complement.csv"))
        test_data = str_to_dict_eedi_df(pd.read_csv("data/zero_shot_more_than_10.csv"))
        # train_data = str_to_dict_eedi_df(pd.read_csv("data/ft_more_than_10_complement.csv"))
        # test_data = str_to_dict_eedi_df(pd.read_csv("data/ft_more_than_10.csv"))
        all_data = pd.concat([train_data, test_data])
        return all_data


def get_prompt(row: pd.Series):
    return "A teacher assigns the following math question to a class of middle school students.\n\n" +\
        f"Question: {row['question']}\n" +\
        f"Solution: {row['correct_option']['explanation']}\n" +\
        f"Correct answer: {row['correct_option']['option']}\n" +\
        "Generate a distractor for this question that targets some student misconception."

def get_label(distractor: dict, args):
    result = "\n"
    if args.gen_feedback:
        result += f"Explanation: {distractor['explanation']}\n"
    result += f"Distractor: {distractor['option']}"
    return result

class StandardDataset(Dataset):
    def __init__(self, df: pd.DataFrame, test: bool, args):
        self.data = [
            {
                "prompt": get_prompt(row),
                "label": None if test else get_label(row["generated_distractors"][i], args),
                "meta_data": row.to_dict()
            }
            for _, row in df.iterrows()
            for i in range(1 if test else 10)
        ]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

class StandardCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[dict]):
        all_prompts = [sample["prompt"] for sample in batch]
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
        if not batch[0].get("label"):
            inputs_tokenized = prompts_tokenized
            labels = None
        else:
            all_inputs = [sample["prompt"] + sample["label"] + self.tokenizer.eos_token for sample in batch]
            inputs_tokenized = self.tokenizer(all_inputs, return_tensors="pt", padding=True).to(device)
            prompt_lens = prompts_tokenized.attention_mask.sum(dim=1)
            labels = inputs_tokenized.input_ids.clone()
            padding_mask = torch.arange(labels.shape[1]).repeat(labels.shape[0], 1).to(device) < prompt_lens.unsqueeze(1)
            labels[padding_mask] = -100
            labels = labels.masked_fill(inputs_tokenized.attention_mask == 0, -100)
        
        return {
            "input_ids": inputs_tokenized.input_ids,
            "attention_mask": inputs_tokenized.attention_mask,
            "labels": labels,
            "meta_data": batch
        }
        
def get_standard_dataloader(data: pd.DataFrame, tokenizer: AutoTokenizer, test: bool, shuffle: bool, args):
    dataset = StandardDataset(data, test, args)
    collator = StandardCollator(tokenizer)
    return DataLoader(dataset, collate_fn=collator, batch_size=args.batch_size, shuffle=shuffle)


def get_over_gen_distractors_log_probs(model, batch):
    batch_size = len(batch["meta_data"])
    cel = torch.nn.CrossEntropyLoss(reduction="none") 
    model_outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    labels = batch["labels"][:, 1:].contiguous().view(-1)
    logits = model_outputs.logits[:, :-1].contiguous().view(labels.shape[0], -1)
    log_probs = -cel(logits, labels)
    log_probs = log_probs.view(batch_size, -1)
    log_probs = log_probs.view(batch_size, -1).sum(-1)
    return log_probs.detach().tolist()

def ranking(args):
    assert args.model_name
    model, tokenizer = get_model(args.base_model, args.model_name, None, True)
    model.eval()
    test_data = get_data(args)
    test_dataloader = get_standard_dataloader(test_data, tokenizer, False, False, args)
    results = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            log_probs = get_over_gen_distractors_log_probs(model, batch)
            results.extend(log_probs)
    results = np.reshape(results, (-1, 10))
    test_data["log_probs"] = results.tolist()
    test_data.to_csv(f"ranking_results/zero_shot_{args.model_name}.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    # test_data.to_csv(f"ranking_results/ft_{args.model_name}.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)


def main():
    initialize_seeds(221)
    print(device)
    parser = argparse.ArgumentParser()
    # Modes
    parser.add_argument("--ranking", action="store_true", help="Evaluate pairwise ranking performance on test set")
    # Settings
    parser.add_argument("--gen_feedback", action="store_true", help="Use feedback as chain of thought for distractor generation")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1", help="Pre-trained base model path")
    parser.add_argument("--model_name", type=str, help="Name of model to save for training or load for testing")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    ranking(args)


if __name__ == "__main__":
    main()