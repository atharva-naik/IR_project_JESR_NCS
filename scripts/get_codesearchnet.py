#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import tqdm
import datasets

dataset_dict = datasets.load_dataset("code_x_glue_tc_nl_code_search_adv") 
for split in dataset_dict:
    dataset = dataset_dict[split]
    json_data = []
    path = f"data/codesearchnet-{split}.json"
    for item in tqdm.tqdm(dataset, desc=f"processing {split}"):
        json_data.append({
            "intent": item["docstring"],
            "snippet": item["code"],
            "question_id": item["id"],
            "prob": item["score"],
        })
    print(f"saving {split} data to: {path}")
    with open(path, "w") as f:
        json.dump(json_data, f, indent=4)