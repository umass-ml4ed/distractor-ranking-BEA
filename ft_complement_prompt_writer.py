import ast
import pandas as pd
import json
from utils import str_to_dict_eedi_df

question_num = 10
# question_num = 3
data = pd.read_csv(f"data/ft_less_than_{question_num}.csv")
data = str_to_dict_eedi_df(data)
prompts = []
instructions = "You are provided with a math question, correct answer, and the explanation of correct answer. Your task is to generate 3 unique incorrect answers (distractors) to be used as multiple-choice options for a middle school math multiple-choice question. Before generating each distractor, include a concise explanation for students to clarify why that is not the correct answer. Ensure each distractor is different from the correct answer and distinct from the others; this is very important!\n"

for idx, row in data.iterrows():
        prompt = f"{instructions}Question: {row['question'].strip()}\nExplanation: {row['correct_option']['explanation'].strip()}\nAnswer: {row['correct_option']['option'].strip()}"
        prompts.append(prompt)

print(len(prompts))
print(prompts[0])
result_address = f"./prompts/second_round_{question_num}.json"
with open(result_address, "w") as f:
    json.dump(prompts, f, indent=4)
        

    
    