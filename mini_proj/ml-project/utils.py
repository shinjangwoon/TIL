import json


def load_config(model_name):
    with open(f'./config/{model_name}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data
