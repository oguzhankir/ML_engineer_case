import json
import os


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_config():
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'artifacts', 'config.json')

    with open(config_path) as json_file:
        data = json.load(json_file)

    return data