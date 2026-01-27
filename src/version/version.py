import argparse
import json
import os
from pathlib import Path
import re


class Version:
    def __init__(self, input_version_datastore):
        self.input_version_datastore = Path(input_version_datastore)

    def next_version(self, version: str) -> str:
        """
        Given a version string like 'v1', 'v1.2', or 'v1.2.3',
        returns the next version by incrementing the last number.
    
        Examples:
            get_next_version("v1")       -> "v2"
            get_next_version("v1.0")     -> "v1.1"
            get_next_version("v2.5.9")   -> "v2.5.10"
            get_next_version("1.0")      -> "1.1"   (also works without 'v')
        """
        # Match optional 'v' prefix and dot-separated numbers
        match = re.fullmatch(r'(v?)(\d+(?:\.\d+)*)', version)
        if not match:
            raise ValueError(f"Invalid version format: {version}")
        
        prefix = match.group(1)  # 'v' or ''
        numbers_str = match.group(2)
        
        # Split into integers
        parts = list(map(int, numbers_str.split('.')))
        
        # Increment the last part
        parts[-1] += 1
        
        # Reconstruct version string
        new_numbers = '.'.join(map(str, parts))
        return prefix + new_numbers

    def fetch_version(self):
        with open(self.input_version_datastore / 'version.json', 'r') as f:
            version = json.load(f)
        return version

    def fetch_next_version(self):
        next_version = self.fetch_version().copy()

        next_version['platinum_version']['dev'] = self.next_version(next_version['platinum_version']['dev'])
        
        tipos = ['inicial_estacional', 'continuidad_estacional',
                 'inicial_regular', 'continuidad_regular']
        
        for tipo in tipos:
            next_version['feats_version'][tipo]['dev'] = self.next_version(next_version['feats_version'][tipo]['dev'])
            next_version['target_version'][tipo]['dev'] = self.next_version(next_version['target_version'][tipo]['dev'])
            next_version['model_version'][tipo]['dev'] = self.next_version(next_version['model_version'][tipo]['dev'])
        
        return next_version

    def fetch_and_save_version(self):
        # _to_github_env
        """
        This function writes the JSON data to GITHUB_ENV file which allows
        passing data between steps in GitHub Actions workflows.
        """
        data = self.fetch_version()
        github_env = os.getenv('GITHUB_ENV')
        if github_env:
            with open(github_env, 'a') as f:
                # Write the JSON as a single-line string to avoid multiline issues
                json_str = json.dumps(data)
                f.write(f"VERSIONS={json_str}\n")
        else:
            print("Warning: GITHUB_ENV not set. Running outside GitHub Actions.")
            print(f"Would have saved: {json.dumps(data, indent=2)}")
    
    def update_platinum_version(self, object:dict, mode:str, new_version:str):
        """
        object: dict
        mode: str
        new_version: str
        
        Returns:
            dict: object
        """
        object[mode] = new_version
        # with open(self.input_version_datastore / 'version.json', 'w') as f:
        #     json.dump(object, f)
        return object

    def update_feats_version(self, object:dict, tipo:str, mode:str, new_version:str):
        """
        object: dict
        tipo: str
        mode: str
        new_version: str
        
        Returns:
            dict: object
        """
        object[tipo][mode] = new_version
        # with open(self.input_version_datastore / 'version.json', 'w') as f:
        #     json.dump(object, f)
        return object

    def update_target_version(self, object:dict, tipo:str, mode:str, new_version:str):
        """
        object: dict
        tipo: str
        mode: str
        new_version: str
        
        Returns:
            dict: object
        """
        object[tipo][mode] = new_version
        # with open(self.input_version_datastore / 'version.json', 'w') as f:
        #     json.dump(object, f)
        return object

    def update_model_version(self, object:dict, tipo:str, mode:str, new_version:str):
        """
        object: dict
        tipo: str
        mode: str
        new_version: str
        
        Returns:
            dict: object
        """
        object[tipo][mode] = new_version
        # with open(self.input_version_datastore / 'version.json', 'w') as f:
        #     json.dump(object, f)
        return object


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_version_datastore", dest='input_version_datastore',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args