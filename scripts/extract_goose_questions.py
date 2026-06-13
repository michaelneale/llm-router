"""Extract genuine user prompts from goose's sessions.db for router training.

Filters:
- role='user', real session types (user/acp/terminal)
- metadata userVisible && agentVisible (drops compaction summaries, synthetic turns)
- drops slash-commands, <info-msg>, <15 chars, >4000 chars
- exact dedupe + near-dupe collapse (normalized prefix)
- tags context-dependent turns ("yes do it", "keep going") so they can be
  excluded or kept via --include-continuations

Output: one question per line (newlines flattened), suitable for
`model-router collect --questions`.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

DB_DEFAULT = Path.home() / ".local/share/goose/sessions/sessions.db"

CONTINUATION = re.compile(
    r"^(yes|yeah|yep|ok|okay|no|nope|sure|go|go on|keep going|continue|do it|"
    r"try again|and |also |then |next |what about|how about|why|hrm|hmm|wait)\b",
    re.IGNORECASE,
)


def extract(db_path: Path, include_continuations: bool) -> tuple[list[str], dict]:
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """
        SELECT m.content_json, m.metadata_json
        FROM messages m JOIN sessions s ON s.id = m.session_id
        WHERE m.role='user' AND s.session_type IN ('user','acp','terminal')
        ORDER BY m.created_timestamp
        """
    ).fetchall()

    stats = {"rows": len(rows), "no_text": 0, "synthetic": 0, "len_filtered": 0, "continuation": 0}
    texts: list[str] = []
    for r in rows:
        try:
            meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            content = json.loads(r["content_json"])
        except (ValueError, TypeError):
            continue
        if not (meta.get("userVisible") and meta.get("agentVisible")):
            stats["synthetic"] += 1
            continue
        t = " ".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
        if not t:
            stats["no_text"] += 1
            continue
        if t.startswith("/") or t.startswith("<info-msg>"):
            stats["synthetic"] += 1
            continue
        if not (15 <= len(t) <= 4000):
            stats["len_filtered"] += 1
            continue
        if not include_continuations and CONTINUATION.match(t) and len(t) < 80:
            stats["continuation"] += 1
            continue
        texts.append(re.sub(r"\s+", " ", t))

    # exact dedupe preserving order, then near-dupe by normalized 80-char prefix
    seen_exact: set[str] = set()
    seen_prefix: set[str] = set()
    out: list[str] = []
    for t in texts:
        if t in seen_exact:
            continue
        seen_exact.add(t)
        p = re.sub(r"[^a-z0-9]", "", t.lower())[:80]
        if p in seen_prefix:
            continue
        seen_prefix.add(p)
        out.append(t)

    stats["kept"] = len(out)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_DEFAULT))
    ap.add_argument("--output", default="data/goose-questions.txt")
    ap.add_argument("--include-continuations", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap output (0 = all)")
    args = ap.parse_args()

    questions, stats = extract(Path(args.db), args.include_continuations)
    if args.limit:
        questions = questions[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(questions) + "\n")

    print(f"DB rows scanned     : {stats['rows']}")
    print(f"  synthetic/hidden  : {stats['synthetic']}")
    print(f"  tool-result only  : {stats['no_text']}")
    print(f"  length-filtered   : {stats['len_filtered']}")
    print(f"  short continuations: {stats['continuation']}")
    print(f"  kept (unique)     : {stats['kept']}")
    print(f"Wrote {len(questions)} questions -> {out}")


if __name__ == "__main__":
    main()
