import json
import re
import pandas as pd
import ast
import csv
from utils import complement_process
import numpy as np

question_num = 10
# question_num = 3
data_address = f"./predictions/ft_second_round_{question_num}.json"

with open(data_address, "r") as f:
    prompt_responses = json.load(f)

distractors = []
distractor_pattern = re.compile(r"(?i)distractor ?(?:1|2|3|4|5|6|7|8|9|1[0-5]): (.+)")

for idx, response in enumerate(prompt_responses):
    distractors_per_question = []
    lines = response.replace("*", "").split("\n")
    for line in lines:
        line = line.strip()
        if distractor_pattern.match(line):
            distractor = distractor_pattern.match(line).group(1).replace("$", "").strip()
            distractor = re.sub(r"([\d\.]+)\s*(/|\\div)\s*([\d\.]+)", r"\\frac{\g<1>}{\g<3>}", distractor)
            distractor = re.sub(r'\s*:\s*',  ':', distractor)
            distractor = re.sub(r'\s*-\s*',  '-', distractor)
            distractors_per_question.append(distractor)
    if len(distractors_per_question) > 3:
        distractors_per_question = distractors_per_question[:3]
    elif len(distractors_per_question) < 3:
        # add placeholder for the missing distractors
        for i in range(3 - len(distractors_per_question)):
            distractors_per_question.append(f"placeholder_{i+1}")
    distractors.append(distractors_per_question)
    
if question_num == 10:
    distractors = np.reshape(distractors, (-1, 15)).tolist()

else:
    distractors = np.reshape(distractors, (-1, 6)).tolist()
    
# always put placeholder at the end of the list
for idx, distractor in enumerate(distractors):
    for element in distractor:
        if "placeholder" in element:
            distractors[idx].remove(element)
            distractors[idx].append(element)
        
non_repeat_distractors = []
for distractors_per_question in distractors:
    non_repeat_distractors_per_question = []
    for element in distractors_per_question:
        if element not in non_repeat_distractors_per_question:
            non_repeat_distractors_per_question.append(element)
    non_repeat_distractors.append(non_repeat_distractors_per_question)

df = complement_process(non_repeat_distractors, question_num, "ft")
df.to_csv(f"data/ft_more_than_{question_num}_complement.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    

    