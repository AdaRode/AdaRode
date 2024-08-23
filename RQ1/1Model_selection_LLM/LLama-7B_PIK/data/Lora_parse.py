import pandas as pd
import json

name = "./PIK/test_set"
# Read the CSV file
csv_file_path = '{}.csv'.format(name)
json_file_path = '{}.json'.format(name)
df = pd.read_csv(csv_file_path)

# Create a list to store the JSON objects
json_list = []

# Process each row in the DataFrame
for index, row in df.iterrows():
    json_object = {
        "instruction": "",
        "input": row['Text'],
        "output": str(row['Label'])
    }
    json_list.append(json_object)

# Write the list of JSON objects to a JSON file

with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_list, json_file, ensure_ascii=False, indent=4)

print(f"JSON file has been successfully created at {json_file_path}")
