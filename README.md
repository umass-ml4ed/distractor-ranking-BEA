# [Improving Automated Distractor Generation for Math Multiple-choice Questions with Overgenerate-and-rank]()

In this repository, we present the code to our paper "Improving Automated Distractor Generation for Math Multiple-choice Questions with Overgenerate-and-rank" by Alexander Scarlatos, Wanyong Feng, Digory Smith, Simon Woodhead, and Andrew Lan. In this work, we propose a novel method to enhance the quality of generated distractors through overgenerate-and-rank, training a ranking model to predict how likely distractors are to be selected by real students. The paper is accepted as the short paper of BEA workshop, NAACL 2024.

For any questions please [email](mailto:wanyongfeng@umass.edu) or raise an issue.

## Running

### Setup
```
python3 -m venv fb_env
source fb_env/bin/activate
python3 -m pip install -r requirements.txt
```

### Train SFT Model

### Train DPO Models

### Generate distractors (zero-shot)
```
