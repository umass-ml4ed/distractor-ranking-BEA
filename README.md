# [Improving Automated Distractor Generation for Math Multiple-choice Questions with Overgenerate-and-rank](https://aclanthology.org/2024.bea-1.19/)

In this repository, we present the code to our paper "Improving Automated Distractor Generation for Math Multiple-choice Questions with Overgenerate-and-rank" by Alexander Scarlatos, Wanyong Feng, Digory Smith, Simon Woodhead, and Andrew Lan. In this work, we propose a novel method to enhance the quality of generated distractors through overgenerate-and-rank, training a ranking model to predict how likely distractors are to be selected by real students. The paper is accepted as a short paper of the BEA workshop at NAACL 2024.

For any questions please [email](mailto:wanyongfeng@umass.edu) or raise an issue.

## Running

### Setup
```
python3 -m venv disrank_env
source disrank_env/bin/activate
python3 -m pip install -r requirements.txt
```

### Ranking Model
#### Train SFT Model
```
python ranking_model_train.py --finetune --model_name disrank-sft
```

#### Train DPO Model
```
python ranking_model_train.py --dpo --model_name disrank-dpo --pt_model_name disrank-sft
```

#### Evaluate Ranking Accuracy
```
python ranking_model_train.py --ranking --model_name disrank-dpo
python ranking_model_train.py --analyze_ranking --model_name disrank-dpo
```

### Generate Distractors (CoT)
```
python zero_shot_prompt_writer.py
python prompting.py
python zero_shot_prediction_processing.py
python zero_shot_complement_prompt_writer.py
python prompting.py
python zero_shot_prediction_complement_processing.py
```

### Generate Distractors (FT)
#### Train Mistral
```
python train.py
```
#### Generate Distractors
```
python test.py
python ft_prediction_processing.py
python ft_complement_prompt_writer.py
python test.py
python ft_prediction_complement_processing.py
```

### Generate Rankings
```
python ranking.py --base_model mistralai/Mistral-7B-v0.1 --model_name xxx --batch_size 16
```

## Citation
If you used our code or found this work useful in any way, please cite us!
```
@inproceedings{scarlatos-etal-2024-improving,
    title = "Improving Automated Distractor Generation for Math Multiple-choice Questions with Overgenerate-and-rank",
    author = "Scarlatos, Alexander  and
      Feng, Wanyong  and
      Lan, Andrew  and
      Woodhead, Simon  and
      Smith, Digory",
    editor = {Kochmar, Ekaterina  and
      Bexte, Marie  and
      Burstein, Jill  and
      Horbach, Andrea  and
      Laarmann-Quante, Ronja  and
      Tack, Ana{\"\i}s  and
      Yaneva, Victoria  and
      Yuan, Zheng},
    booktitle = "Proceedings of the 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.bea-1.19",
    pages = "222--231",
}
```
