import json
import re
import csv
from utils import process

question_num = 10
# question_num = 3
data_address = f"./predictions/zero_shot_{question_num}_predictions.json"
with open(data_address, "r") as f:
    prompt_responses = json.load(f)

distractors = []

distractor_pattern = re.compile(r"(?i)distractor ?(?:1|2|3|4|5|6|7|8|9|1[0-5]): (.+)")

for idx, response in enumerate(prompt_responses):
    distractors_per_question = []
    lines = response.replace("*", "").split("\n")
    for line in lines:
        if distractor_pattern.match(line):
            distractor = distractor_pattern.match(line).group(1).split("(repeated")[0].split("(Note")[0].split(' (incorrectly')[0].split(', but with a note')[0].split(", with a note")[0].split( "(but")[0].split(', but rounded')[0].split(' (or')[0].split(" (approximately)")[0].split(' (perimeter assumption)')[0].split(' (Non-simplified')[0]
            distractor = distractor.split(' (16 sides)')[0].split('(not specifying')[0].split(' (misunderstanding')[0].split('(not recognizing')[0].split(' (ambiguous,')[0].split('(assuming a circle')[0].split('(Different reasoning from Distractor3)')[0].split(', because')[0].split(' (as')[0].split('or equivalently')[0].split('(doubled')[0]
            distractor = distractor.split('(16-side')[0].split('(achieved')[0].split('(9 + 9')[0].split('(9 x 4')[0].split(' (This')[0].strip()
            distractors_per_question.append(distractor)
    
    # remove the repeated distractors, keep the order of the distractors
    no_repeat_distractors_per_question = []
    for distractor in distractors_per_question:
        if distractor not in no_repeat_distractors_per_question:
            no_repeat_distractors_per_question.append(distractor)
    distractors.append(no_repeat_distractors_per_question)

qualified_df, unqualified_df = process(distractors, question_num, "zero_shot")
    
qualified_df.to_csv(f"data/zero_shot_more_than_{question_num}.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
unqualified_df.to_csv(f"data/zero_shot_less_than_{question_num}.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    

    