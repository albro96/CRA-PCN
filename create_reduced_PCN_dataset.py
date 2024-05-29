import json
import os

dataset_json_path = "CRA-PCN/datasets/ShapeNet.json"
dataset_json_reduced_path = "CRA-PCN/datasets/ShapeNet_airplane.json"

with open(dataset_json_path, "r") as f:
    dataset = json.load(f)

for dataset_dict in dataset:
    with open(
        dataset_json_path.replace(".json", f'_{dataset_dict["taxonomy_name"]}.json'),
        "w",
    ) as f:
        json.dump([dataset_dict], f, indent=4)
