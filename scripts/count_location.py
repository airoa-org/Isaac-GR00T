import json

json_path = "/home/group_25b505/group_6/workspace/user_00085_25b505/projects/Isaac-GR00T/demo_data/2025-05-v3.0-success-only/meta/episodes.jsonl"

with open('hoge.jsonl') as f:
    jsonl_data = [json.loads(l) for l in f.readlines()]