import json 

with open('dataset.json', 'r') as f:
    data = json.load(f)

alpaca_data = []
# Convert to prompt-response format
for item in data:
    alpaca_entry = { 
        "instruction": item['prompt'],
        "input": "",
        "output": item['response'],
        "id": item['id']
    } 
    alpaca_data.append(alpaca_entry)

with open('convert_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(alpaca_data, f, indent=2)
    
    print(f"Converted {len(alpaca_data)} entries to Alpaca format")
    print(f"Saved to convert_dataset.json")