import json

file_path = r"C:\Users\Reeth Jain\.cache\huggingface\hub\datasets--McAuley-Lab--Amazon-Reviews-2023\snapshots\2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e\raw\review_categories\All_Beauty.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        record = json.loads(line)
        print(record)

        if i == 4:
            break