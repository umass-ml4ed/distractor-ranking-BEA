from utils import str_to_dict_eedi_df
import pandas as pd
import json

def process_data(data_address):
    data_file = pd.read_csv(data_address)
    data_file = str_to_dict_eedi_df(data_file)
    data = []
    for _, row in data_file.iterrows():
        temp = {}
        input = "You are provided with a math question, correct answer, and the explanation of correct answer. Your task is to generate 3 unique incorrect answers (distractors) to be used as multiple-choice options for a middle school math multiple-choice question. Before generating each distractor, include a concise explanation for students to clarify why that is not the correct answer. Ensure each distractor is different from the correct answer and distinct from the others; this is very important!\n"
        input = input + "Question: " + row["question"].strip() + "\nExplanation: " + row["correct_option"]["explanation"].strip() + " \nAnswer: " + row["correct_option"]["option"].strip()
        temp["input"] = input
        output = "\nDistractor1 Feedback: " + row["distractors"][0]["explanation"].strip() + "\nDistractor1: " + row["distractors"][0]["option"].strip() 
        output += "\nDistractor2 Feedback: " + row["distractors"][1]["explanation"].strip() + "\nDistractor2: " + row["distractors"][1]["option"].strip()
        output += "\nDistractor3 Feedback: " + row["distractors"][2]["explanation"].strip() + "\nDistractor3: " + row["distractors"][2]["option"].strip()
        temp["output"] = output
        data.append(temp)
    return data

test_data = process_data("data/")
temp_data = process_data("data/")
train_data = temp_data[:int(len(temp_data)*0.8)]
valid_data = temp_data[int(len(temp_data)*0.8):]

with open('data/train.json', 'w') as outfile:
    json.dump(train_data, outfile, indent=4)
        
with open('data/valid.json', 'w') as outfile:
    json.dump(valid_data, outfile, indent=4)

with open('data/test.json', 'w') as outfile:
    json.dump(test_data, outfile, indent=4)