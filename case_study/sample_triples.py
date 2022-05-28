import json
import random
import pandas as pd

def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))
            
    return data

def sample_dict(d: dict, k: int=200, seed=42):
    random.seed(seed)
    sampled_keys = random.sample(list(d.keys()), k=k)
    sampled_dict = {k: d[k] for k in sampled_keys}
    
    return sampled_dict

def filter_dict_by_num_values(d: dict, num_values: int=2):
    return {k: v for k,v in d.items() if len(v) >= num_values}
    
triples = read_jsonl("../data/conala-mined.jsonl")
intent_rec_map = {}
for rec in triples:
    try: intent_rec_map[rec["intent"]].append(rec)
    except KeyError: intent_rec_map[rec["intent"]] = [rec]
# filter intent with at least 2 values/code snippets assoicated with it.
filt_intent_rec_map = filter_dict_by_num_values(
    intent_rec_map, 
    num_values=2
)
# sample 200 random values.
sampled_posts = sample_dict(filt_intent_rec_map, k=200)
df = []
for intent, rec_list in sampled_posts.items():
    row = {
        "intent": intent,
        "code L": rec_list[0]["snippet"],
        "code R": rec_list[1]["snippet"],
        "L > R": int(rec_list[0]["prob"] > rec_list[1]["prob"]),
        "rel L": rec_list[0]["prob"],
        "rel R": rec_list[1]["prob"],
    }
    df.append(row)
df = pd.DataFrame(df)
print(df)
df.to_csv("case_study_triples.csv")
print(df["L > R"].tolist())
print()