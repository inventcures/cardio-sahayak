import json

def fix_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    decoder = json.JSONDecoder()
    pos = 0
    records = []
    
    while pos < len(content):
        # Skip whitespace
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
            
        try:
            obj, end = decoder.raw_decode(content, pos)
            records.append(obj)
            pos = end
        except json.JSONDecodeError as e:
            print(f"Error decoding at pos {pos}: {e}")
            break
            
    print(f"Recovered {len(records)} records from {filepath}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

fix_jsonl("data/processed_datasets/cardio_sahayak_india_instruct_v2.jsonl")
