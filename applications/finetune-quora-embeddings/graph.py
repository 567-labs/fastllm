import json
import csv

with open("optimize.json") as file:
    data = json.load(file)

keys = set()
for obj in data:
    keys.update(obj.keys())


with open("prev_results.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=keys)
    writer.writeheader()
    writer.writerows(data)
