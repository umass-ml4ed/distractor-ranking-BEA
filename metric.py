import pandas as pd
import ast
import re
from utils import initialize_seeds
import random
from utils import str_to_dict_eedi_df, clean_string, relaxed_metric, hard_metric, weighted_proportional_metric, proportional_metric


def main():
    initialize_seeds(40)
    questions = []
    gt_distractors = []
    generated_distractors = []
    generated_uncleaned_distractors = []
    log_probs = []
    proportions = []
    
    data = pd.read_csv("ranking_results/zero_shot_disgen-dpo-t0.csv")
    # data = pd.read_csv("ranking_results/ft_disgen-dpo-t0.csv")
    data = str_to_dict_eedi_df(data)
    
    for idx, row in data.iterrows():
        questions.append(row["question"].strip())
        gt_distractors.append([clean_string(distractor['option']) for distractor in row["gt_distractors"]])
        # save as a list of dictionaries, where the key is the option and the value is the proportion
        proportions.append({clean_string(distractor['option']) : distractor['proportion'] for distractor in row["gt_distractors"]})
        generated_distractors.append([clean_string(distractor['option']) for distractor in row["generated_distractors"]])
        generated_uncleaned_distractors.append([distractor['option'] for distractor in row["generated_distractors"]])
        # generated_distractors.append([distractor['option'] for distractor in row["generated_distractors"]])
        log_probs.append(row["log_probs"])

    top3_distractors = []
    # sort the distractors based on the log_probs and get the top3 distractors
    for idx, generated_distractor in enumerate(generated_distractors):
        sorted_distractor = sorted(zip(generated_distractor, log_probs[idx]), key=lambda x: x[1], reverse=True)
        top3_distractor = [distractor for distractor, log_prob in sorted_distractor[:3]]
        top3_distractors.append(top3_distractor)
        
    random_3_distractors = []
    # randomly choose 3 distractors from the generated distractors
    for idx, generated_distractor in enumerate(generated_distractors):
        random_3_distractor = random.sample(generated_distractor, 3)
        random_3_distractors.append(random_3_distractor)
    
    print("Relaxed metric for top3 distractors: ", relaxed_metric(gt_distractors, top3_distractors))
    print("Hard metric for top3 distractors: ", hard_metric(gt_distractors, top3_distractors))
    print("Proportional metric for top3 distractors: ", proportional_metric(gt_distractors, top3_distractors))
    print("weighed Proportional metric for top3 distractors: ", weighted_proportional_metric(gt_distractors, top3_distractors, proportions))
    
    
    print("Relaxed metric for random 3 distractors: ", relaxed_metric(gt_distractors, random_3_distractors))
    print("Hard metric for random 3 distractors: ", hard_metric(gt_distractors, random_3_distractors))
    print("Proportional metric for random 3 distractors: ", proportional_metric(gt_distractors, random_3_distractors))
    print("weighed Proportional metric for random 3 distractors: ", weighted_proportional_metric(gt_distractors, random_3_distractors, proportions))
    
if __name__ == "__main__":
    main()


    