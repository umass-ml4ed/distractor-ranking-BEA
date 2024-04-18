from typing import List
import argparse
import re
import json
from itertools import combinations, permutations
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils import initialize_seeds, device, str_to_dict_eedi_df, get_model, ref_model_ctm

pred_re = re.compile(r"Distractor: (.*)$")

def get_data(fold: int = 0):
    train_data = str_to_dict_eedi_df(pd.read_csv("data/eedi_train_80_cleaned_4_18.csv"))
    test_data = str_to_dict_eedi_df(pd.read_csv("data/eedi_test_20_cleaned_4_18.csv"))
    all_data = pd.concat([train_data, test_data])
    # all_data = all_data[:20]
    split_point = int((fold / 5) * len(all_data))
    all_data = pd.concat([all_data[split_point:], all_data[:split_point]])
    return (
        all_data[:int(.64 * len(all_data))],
        all_data[int(.64 * len(all_data)):int(.8 * len(all_data))],
        all_data[int(.8 * len(all_data)):]
    )

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

def get_first_win_prob(prop1: float, prop2: float, args):
    if prop1 == prop2: # Handles when both are 0
        return .5
    if args.temp == 0:
        return 1 if prop1 > prop2 else 0
    exp = 1 / args.temp
    return prop1 ** exp / (prop1 ** exp + prop2 ** exp)

def get_labels_tensor(prompts_tokenized, inputs_tokenized):
    prompt_lens = prompts_tokenized.attention_mask.sum(dim=1)
    labels = inputs_tokenized.input_ids.clone()
    padding_mask = torch.arange(labels.shape[1]).repeat(labels.shape[0], 1).to(device) < prompt_lens.unsqueeze(1)
    labels[padding_mask] = -100
    labels = labels.masked_fill(inputs_tokenized.attention_mask == 0, -100)
    return labels

class StandardDataset(Dataset):
    def __init__(self, df: pd.DataFrame, test: bool, args):
        self.data = [
            {
                "prompt": get_prompt(row),
                "label": None if test else get_label(row["distractors"][i], args),
                "meta_data": row.to_dict()
            }
            for _, row in df.iterrows()
            for i in range(1 if test else 3)
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
            labels = get_labels_tensor(prompts_tokenized, inputs_tokenized)

        return {
            "input_ids": inputs_tokenized.input_ids,
            "attention_mask": inputs_tokenized.attention_mask,
            "labels": labels,
            "meta_data": batch
        }

class PairwiseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, args):
        self.data = [
            {
                "prompt": get_prompt(row),
                "labels": [
                    get_label(row["distractors"][idx0], args),
                    get_label(row["distractors"][idx1], args)
                ],
                "first_win_prob": get_first_win_prob(
                    row["distractors"][idx0]["proportion"], row["distractors"][idx1]["proportion"], args),
                "options": [row["distractors"][idx0]["option"], row["distractors"][idx1]["option"]],
                "proportions": [row["distractors"][idx0]["proportion"], row["distractors"][idx1]["proportion"]],
                "meta_data": row.to_dict()
            }
            for _, row in df.iterrows()
            for idx0, idx1 in combinations(range(3), 2)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

class PairwiseCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[dict]):
        all_prompts = [sample["prompt"] for sample in batch]
        all_prompts = all_prompts * 2
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
        first_completions = [sample["prompt"] + sample["labels"][0] + self.tokenizer.eos_token for sample in batch]
        second_completions = [sample["prompt"] + sample["labels"][1] + self.tokenizer.eos_token for sample in batch]
        all_inputs = first_completions + second_completions
        inputs_tokenized = self.tokenizer(all_inputs, return_tensors="pt", padding=True).to(device)
        labels = get_labels_tensor(prompts_tokenized, inputs_tokenized)

        return {
            "input_ids": inputs_tokenized.input_ids,
            "attention_mask": inputs_tokenized.attention_mask,
            "labels": labels,
            "first_win_probs": torch.Tensor([sample["first_win_prob"] for sample in batch]).to(device),
            "meta_data": batch
        }

class DisGroupedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, args):
        self.data = [
            {
                "prompt": get_prompt(row),
                "labels": [get_label(dis, args) for dis in row["distractors"]],
                "proportions": [dis["proportion"] for dis in row["distractors"]],
                "meta_data": row.to_dict()
            }
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

class DisGroupedCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[dict]):
        all_prompts = [sample["prompt"] for sample in batch for _ in range(3)]
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
        all_inputs = [sample["prompt"] + label + self.tokenizer.eos_token for sample in batch for label in sample["labels"]]
        inputs_tokenized = self.tokenizer(all_inputs, return_tensors="pt", padding=True).to(device)
        labels = get_labels_tensor(prompts_tokenized, inputs_tokenized)

        return {
            "input_ids": inputs_tokenized.input_ids,
            "attention_mask": inputs_tokenized.attention_mask,
            "labels": labels,
            "proportions": torch.Tensor([sample["proportions"] for sample in batch]).to(device),
            "meta_data": batch
        }

def get_standard_dataloader(data: pd.DataFrame, tokenizer: AutoTokenizer, test: bool, shuffle: bool, args):
    dataset = StandardDataset(data, test, args)
    collator = StandardCollator(tokenizer)
    return DataLoader(dataset, collate_fn=collator, batch_size=args.batch_size, shuffle=shuffle)

def get_pairwise_dataloader(data: pd.DataFrame, tokenizer: AutoTokenizer, shuffle: bool, args):
    dataset = PairwiseDataset(data, args)
    collator = PairwiseCollator(tokenizer)
    return DataLoader(dataset, collate_fn=collator, batch_size=args.batch_size, shuffle=shuffle)

def get_pairwise_log_probs(model, batch):
    batch_size = len(batch["meta_data"])
    cel = torch.nn.CrossEntropyLoss(reduction="none")
    labels = batch["labels"][:, 1:].contiguous().view(-1)
    model_outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = model_outputs.logits[:, :-1].contiguous().view(labels.shape[0], -1)
    log_probs = -cel(logits, labels)
    log_probs = log_probs.view(batch_size * 2, -1).sum(-1) # Take likelihood, mean would be -perplexity
    return log_probs[:batch_size], log_probs[batch_size:]

def ft_loss(model, batch, _args):
    model_outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    return model_outputs.loss

def dpo_loss(model, batch, args):
    bcel = torch.nn.BCEWithLogitsLoss()
    first_log_probs, second_log_probs = get_pairwise_log_probs(model, batch)
    with torch.no_grad():
        with ref_model_ctm(model):
            pt_first_log_probs, pt_second_log_probs = get_pairwise_log_probs(model, batch)
    first_wins_logits = args.beta * (
        (first_log_probs - pt_first_log_probs) -
        (second_log_probs - pt_second_log_probs)
    )
    return bcel(first_wins_logits, batch["first_win_probs"])

def train(train_dataloader: DataLoader, val_dataloader: DataLoader, model: AutoModelForCausalLM, loss_fn, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = None
    for epoch in range(args.epochs):
        total_train_loss = 0
        total_val_loss = 0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            loss = loss_fn(model, batch, args)
            total_train_loss += loss.item()
            loss = loss / args.grad_accum_steps
            loss.backward()
            if (step + 1) % args.grad_accum_steps == 0 or step == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader):
                loss = loss_fn(model, batch, args)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if not best_val_loss or avg_val_loss < best_val_loss:
            print("Best! Saving model...")
            model.save_pretrained(args.model_name)
            best_val_loss = avg_val_loss

def finetune(args, fold: int = 0):
    assert args.model_name
    model, tokenizer = get_model(args.base_model, None, None, False)
    train_data, val_data, _ = get_data(fold)
    train_dataloader = get_standard_dataloader(train_data, tokenizer, False, True, args)
    val_dataloader = get_standard_dataloader(val_data, tokenizer, False, False, args)
    train(train_dataloader, val_dataloader, model, ft_loss, args)

def dpo(args, fold: int = 0):
    assert args.model_name
    assert args.pt_model_name
    model, tokenizer = get_model(args.base_model, None, args.pt_model_name, False)
    train_data, val_data, _ = get_data(fold)
    train_dataloader = get_pairwise_dataloader(train_data, tokenizer, True, args)
    val_dataloader = get_pairwise_dataloader(val_data, tokenizer, False, args)
    train(train_dataloader, val_dataloader, model, dpo_loss, args)

def generate_beams(model, batch, tokenizer, args):
    batch_gen_dis = [set() for _ in range(len(batch["meta_data"]))]
    output_ids = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_gen_tokens,
        num_return_sequences=args.num_samples,
        do_sample=False,
        num_beams=args.num_samples,
    )
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    for i, pred in enumerate(preds):
        pred_match = pred_re.search(pred)
        if pred_match:
            batch_gen_dis[i // args.num_samples].add(pred_match.group(1))
    return batch_gen_dis

def generate_samples(model, batch, tokenizer, args):
    batch_size = len(batch["meta_data"])
    batch_gen_dis = [set() for _ in range(batch_size)]
    enough_samples = np.zeros(batch_size, dtype=bool)
    max_tries = 3 * args.num_samples
    cur_try = 0
    while not enough_samples.all() and cur_try < max_tries:
        cur_try += 1
        output_ids = model.generate(
            input_ids=batch["input_ids"][~enough_samples],
            attention_mask=batch["attention_mask"][~enough_samples],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_gen_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        pred_i = 0
        for i in range(batch_size):
            if not enough_samples[i]:
                pred = preds[pred_i]
                pred_i += 1
                pred_match = pred_re.search(pred)
                if pred_match:
                    batch_gen_dis[i].add(pred_match.group(1))
                    if len(batch_gen_dis[i]) == args.num_samples:
                        enough_samples[i] = True
    return batch_gen_dis

def generate(args, fold: int = 0):
    assert args.model_name
    model, tokenizer = get_model(args.base_model, args.model_name, None, True)
    model.eval()
    tokenizer.padding_side = "left"
    _, _, test_data = get_data(fold)
    test_dataloader = get_standard_dataloader(test_data, tokenizer, True, False, args)

    total_matched = 0
    total_partial = 0
    total_exact = 0
    results = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            if args.decoding == "beam":
                batch_gen_dis = generate_beams(model, batch, tokenizer, args)
            else:
                batch_gen_dis = generate_samples(model, batch, tokenizer, args)
            batch_gold_dis = [
                [dis["option"] for dis in sample["meta_data"]["distractors"]]
                for sample in batch["meta_data"]
            ]
            batch_matched = np.array([
                len(set(gold_dis) & gen_dis)
                for gold_dis, gen_dis in zip(batch_gold_dis, batch_gen_dis)
            ])
            total_matched += batch_matched.sum()
            total_partial += (batch_matched > 0).sum()
            total_exact += (batch_matched == 3).sum()
            for sample, gold_dis, gen_dis, matched in zip(batch["meta_data"], batch_gold_dis, batch_gen_dis, batch_matched):
                results.append({
                    "question": sample["meta_data"]["question"],
                    "gold_distractors": gold_dis,
                    "pred_distractors": list(gen_dis),
                    "matched": int(matched)
                })

    prop = 100 * total_matched / (len(test_data) * 3)
    partial = 100 * total_partial / len(test_data)
    exact = 100 * total_exact / len(test_data)
    with open(f"results/disgen_results_{args.model_name}_{args.decoding}_n{args.num_samples}.json", "w", encoding="utf-8") as f:
        json.dump({
            "proportional": prop,
            "partial": partial,
            "exact": exact,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Proportional: {prop:.2f}, Partial: {partial:.2f}, Exact: {exact:.2f}")
    return prop, partial, exact

def ranking(args, fold: int = 0):
    assert args.model_name
    model, tokenizer = get_model(args.base_model, args.model_name, None, True)
    model.eval()
    _, _, test_data = get_data(fold)
    test_dataloader = get_pairwise_dataloader(test_data, tokenizer, False, args)

    total_correct = 0
    total_pairs = 0
    total_mse = 0
    results = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch_size = len(batch["meta_data"])
            total_pairs += batch_size
            first_log_probs, second_log_probs = get_pairwise_log_probs(model, batch)
            p_first_wins = torch.sigmoid(first_log_probs - second_log_probs)
            first_wins_pred = p_first_wins > .5
            first_wins_label = batch["first_win_probs"] > .5
            batch_correct = first_wins_pred == first_wins_label
            total_correct += batch_correct.sum().item()
            batch_mse = (p_first_wins - batch["first_win_probs"]) ** 2
            total_mse += batch_mse.sum().item()
            for i in range(batch_size):
                results.append({
                    "question": batch["meta_data"][i]["meta_data"]["question"],
                    "options": batch["meta_data"][i]["options"],
                    "proportions": batch["meta_data"][i]["proportions"],
                    "first_wins_label": batch["first_win_probs"][i].item(),
                    "first_wins_pred": p_first_wins[i].item(),
                    "mse": batch_mse[i].item(),
                    "correct": batch_correct[i].item(),
                })

    accuracy = 100 * total_correct / total_pairs
    avg_mse = total_mse / total_pairs
    with open(f"results/ranking_results_{args.model_name}.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "mse": avg_mse,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Accuracy: {accuracy:.2f}, MSE: {avg_mse:.4f}")
    return accuracy, avg_mse

def analyze_ranking(args):
    with open(f"results/ranking_results_{args.model_name}.json") as f:
        results = json.load(f)["results"]
    cutoff_to_total = {cutoff: 0 for cutoff in [0, 3, 5, 10, 15, 20]}
    cutoff_to_correct = cutoff_to_total.copy()
    for result in results:
        for cutoff in cutoff_to_total:
            if abs(result["proportions"][0] - result["proportions"][1]) >= cutoff:
                cutoff_to_total[cutoff] += 1
                if result["correct"]:
                    cutoff_to_correct[cutoff] += 1
    for cutoff, total in cutoff_to_total.items():
        print(f"{cutoff} - Correct: {100 * cutoff_to_correct[cutoff] / total:.2f}%, Portion: {100 * total / len(results):.2f}%")

def crossval(args):
    assert args.finetune or args.dpo
    model_name = args.model_name
    pt_model_name = args.pt_model_name
    results = []
    for fold in range(5):
        print(f"\nFold {fold}")
        args.model_name = f"{model_name}-{fold}"
        if args.finetune:
            finetune(args, fold)
        elif args.dpo:
            args.pt_model_name = f"{pt_model_name}-{fold}"
            dpo(args, fold)
        prop, partial, exact = generate(args, fold)
        accuracy, mse = ranking(args, fold)
        results.append([partial, exact, prop, accuracy, mse])
    results = np.array(results)
    means = results.mean(axis=0)
    stds = results.std(axis=0)
    print("Partial & Exact & Proportional & Accuracy & MSE")
    print(" & ".join([f"${mean:.2f} \\pm {std:.2f}$" for mean, std in zip(means, stds)]))

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    # Modes
    parser.add_argument("--finetune", action="store_true", help="Supervised finetuning for distractor generation")
    parser.add_argument("--dpo", action="store_true", help="DPO training (with student preferences) for distractor generation")
    parser.add_argument("--generate", action="store_true", help="Generate distractors for test set")
    parser.add_argument("--ranking", action="store_true", help="Evaluate pairwise ranking performance on test set")
    parser.add_argument("--analyze_ranking", action="store_true", help="Compute additional statistics on ranking results")
    parser.add_argument("--crossval", action="store_true", help="Cross-validation experiment; includes training (specify finetune or dpo) followed by generate and ranking")
    # Settings
    parser.add_argument("--gen_feedback", action="store_true", help="Use feedback as chain of thought for distractor generation")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1", help="Pre-trained base model path")
    parser.add_argument("--model_name", type=str, help="Name of model to save for training or load for testing")
    parser.add_argument("--pt_model_name", type=str, help="Name of pre-trained (SFT) model for DPO training")
    parser.add_argument("--beta", type=float, default=0.5, help="KL regularization coefficient for DPO training")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for preference probability sharpness")
    parser.add_argument("--decoding", type=str, choices=["beam", "sample"], default="beam", help="Decoding strategy for generation")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of distractors to sample per question during generation")
    parser.add_argument("--max_gen_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    args = parser.parse_args()

    if args.crossval: # Has to go first since done in conjunction with other modes
        crossval(args)
    elif args.finetune:
        finetune(args)
    elif args.dpo:
        dpo(args)
    elif args.generate:
        generate(args)
    elif args.ranking:
        ranking(args)
    elif args.analyze_ranking:
        analyze_ranking(args)

if __name__ == "__main__":
    main()
