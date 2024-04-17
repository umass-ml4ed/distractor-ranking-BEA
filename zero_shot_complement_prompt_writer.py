import ast
import pandas as pd
import json
from utils import str_to_dict_eedi_df

question_num = 10
# question_num = 3
data = pd.read_csv(f"data/zero_shot_less_than_{question_num}.csv")
data = str_to_dict_eedi_df(data)
if question_num == 10:
    instruction = f"You are provided with a math question, correct answer, and the explanation of correct answer. Your task is to use the following template to create {question_num + 5} unique incorrect answers (distractors) to be used as multiple-choice options for a middle school math multiple-choice question. Before generating each distractor, include a concise explanation to clarify for students why that is not the correct answer. Make sure each distractor is clearly different from the correct answer and distinct from each other. There are some example distractors given, make sure that the generated distractors are different from example distractors, this is very important, so pay attention!\n[Template]\n"
else:
    instruction = f"You are provided with a math question, correct answer, and the explanation of correct answer. Your task is to use the following template to create {question_num} unique incorrect answers (distractors) to be used as multiple-choice options for a middle school math multiple-choice question. Before generating each distractor, include a concise explanation to clarify for students why that is not the correct answer. Make sure each distractor is clearly different from the correct answer and distinct from each other. There are some example distractors given, make sure that the generated distractors are different from example distractors, this is very important, so pay attention!\n[Template]\n"

template = ""
if question_num == 10:
    for i in range(question_num + 5):
        template += f"Distractor{i+1} Feedback: \nDistractor{i+1}: \n"
else:
    for i in range(question_num):
        template += f"Distractor{i+1} Feedback: \nDistractor{i+1}: \n"
        
instruction += template
prompts = []
for idx, row in data.iterrows():
        prompt = f"{instruction}Question: {row['question'].strip()}\nExplanation: {row['correct_option']['explanation'].strip()}\nAnswer: {row['correct_option']['option'].strip()}"
        generated_distractors = row["generated_distractors"]
        for i, distractor in enumerate(generated_distractors):
            prompt += f"\nExample distractor{i+1}: {distractor['option']}"
        prompts.append(prompt)

print(prompts[0])
result_address = f"./prompts/zero_shot_{question_num}_complement.json"
with open(result_address, "w") as f:
    json.dump(prompts, f, indent=4)
        

    
    