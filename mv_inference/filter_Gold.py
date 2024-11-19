import json

# File paths
input_file = '/home/gjy/jingyu/InstantMesh/filtered_obj_name_cam7.json'   # Replace with your input JSON file path
output_file = '/home/gjy/jingyu/InstantMesh/train_cam7.json' # Replace with your desired output JSON file path

# Load the JSON data from the input file
with open(input_file, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Filter paths ending with '_Gold'
gold_objs = [obj for obj in json_data['all_objs'] if obj.endswith('_Gold')]

# Update the JSON data
json_data['all_objs'] = gold_objs

# Write the updated JSON data to the output file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print(f"Filtered JSON data saved to {output_file}")
