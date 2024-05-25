import json
def get_config():
    with open('src/koc_finans_case/artifacts/config.json') as json_file:
        data = json.load(json_file)
    return data