import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

def normalize(s: str) -> str:
    # 前後空白除去 → 連続空白を1つに → 末尾の句読点や空白を除去 → 小文字化
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(" .")
    return s.lower()

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def main():
    ap = argparse.ArgumentParser(description="Instructionごとにepisode_indexのリストを作る")
    ap.add_argument("--jsonl", required=True, help="episodes.jsonl のパス")
    ap.add_argument("--ep-min", type=int, default=None, help="集計する最小 episode_index（含む）")
    ap.add_argument("--ep-max", type=int, default=None, help="集計する最大 episode_index（含む）")
    ap.add_argument("--out", required=True, help="出力先JSONファイルパス")
    args = ap.parse_args()

    path = Path(args.jsonl)
    assert path.exists(), f"Not found: {path}"

    task_to_indices = defaultdict(list)

    for ep in load_jsonl(path):
        ep_idx = ep.get("episode_index")
        if not isinstance(ep_idx, int):
            continue
        if args.ep_min is not None and ep_idx < args.ep_min:
            continue
        if args.ep_max is not None and ep_idx > args.ep_max:
            continue

        for t in ep.get("tasks", []) or []:
            if isinstance(t, str) and t.strip():
                task_to_indices[normalize(t)].append(ep_idx)

    # 保存
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(task_to_indices, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(task_to_indices)} instructions to {args.out}")

if __name__ == "__main__":
    main()
