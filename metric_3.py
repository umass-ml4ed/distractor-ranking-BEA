import pandas as pd
import ast
import re
from utils import initialize_seeds
import random
from utils import str_to_dict_eedi_df, clean_string, relaxed_metric, hard_metric, proportional_metric, weighted_proportional_metric


def main():
    initialize_seeds(40)
    
    questions = []
    gt_distractors = []
    un_cleaned_gt_distractors = []
    generated_distractors = []
    un_cleaned_generated_distractors = []
    proportions = []
    
    # data1 = pd.read_csv("data/zero_shot_more_than_3_complement.csv")
    data1 = pd.read_csv("data/ft_more_than_3_complement.csv")

    data1 = str_to_dict_eedi_df(data1)
    
    # data2 = pd.read_csv("data/zero_shot_more_than_3.csv")
    data2 = pd.read_csv("data/ft_more_than_3.csv")
    data2 = str_to_dict_eedi_df(data2)
    
    # combine the two dataframes
    data = pd.concat([data1, data2], ignore_index=True)
    
    for idx, row in data.iterrows():
        questions.append(row["question"].strip())
        gt_distractors.append([clean_string(distractor['option']) for distractor in row["gt_distractors"]])
        un_cleaned_gt_distractors.append([distractor['option'] for distractor in row["gt_distractors"]])
        # save as a list of dictionaries, where the key is the option and the value is the proportion
        proportions.append({clean_string(distractor['option']) : distractor['proportion'] for distractor in row["gt_distractors"]})
        generated_distractors.append([clean_string(distractor['option']) for distractor in row["generated_distractors"]])
        un_cleaned_generated_distractors.append([distractor['option'] for distractor in row["generated_distractors"]])

    print("Relaxed metric: ", relaxed_metric(gt_distractors, generated_distractors))
    print("Hard metric: ", hard_metric(gt_distractors, generated_distractors))
    print("Proportional metric: ", proportional_metric(gt_distractors, generated_distractors))
    print("weighed Proportional metric: ", weighted_proportional_metric(gt_distractors, generated_distractors, proportions))

if __name__ == "__main__":
    main()


    