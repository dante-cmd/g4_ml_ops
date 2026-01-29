import json


class Version:
    def __init__(self):
        with open("version.json", "r") as f:
            self.version = json.load(f)

    def get_version(self,folder:str ,model_name:str, alias:str):
        return self.version[folder][model_name][alias]

    def update_version(self,folder:str, model_name:str, alias:str, version:str):
        self.version[folder][model_name][alias] = version
        with open("version.json", "w") as f:
            json.dump(self.version, f, indent=4)


if __name__ == "__main__":
    version = Version()
    # print(version.version)
    print(version.get_version("feats_version","inicial_regular", "champion"))

    print(version.get_version("target_version", "inicial_regular", "champion"))

