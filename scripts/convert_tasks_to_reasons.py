#!/usr/bin/env python3
import json

DATA_PATH = "/home/group_25b505/group_6/workspace/user_00085_25b505/projects/Isaac-GR00T/demo_data/2025-05-v3.0-success-only_reason"
INPUT_PATH = DATA_PATH + "/meta/tasks.jsonl"
OUTPUT_PATH = DATA_PATH + "/meta/reasons.jsonl"

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # 新しいdictを作ってキーをリネーム
            new_obj = {}

            for k, v in obj.items():
                if k == "task_index":
                    new_obj["reason_index"] = v
                elif k == "task":
                    new_obj["reason"] = v
                else:
                    # その他のキーはそのままコピー
                    new_obj[k] = v

            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
