import json

with open("version.json", "r") as f:
    version = json.load(f)

print(version)