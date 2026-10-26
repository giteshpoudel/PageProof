"""P3 smoke test: a few hand-written questions, each with a known gold page, asked
against the clean text and Tesseract at levels 3 and 5.

Not an eval (that is P4, ~200 verified questions): it checks that `ask` works end
to end and shows how answers and citations change with OCR damage.

    uv run python scripts/p3_smoke.py
"""

import json
from pathlib import Path

from pageproof.qa.answer import ask

QUESTIONS = [  # (question, gold page id, words a correct answer contains)
    ("How much money did Wickham receive in lieu of the church living?", "pg1342-p0262", ["three thousand"]),
    ("Where did the man hide the stolen stone, and whose geese were they?", "pg1661-p0219", ["goose", "sister"]),
    ("According to Darwin, how are hooked seeds transported?", "pg1228-p0440", ["wool", "fur"]),
    ("Which companies does Adam Smith say lack any pretext of great utility to justify a joint-stock company?", "pg3300-p1061", ["copper", "glass"]),
    ("What did one of the Masonic Brothers tell Pierre about his treatment of his wife?", "pg2600-p0790", ["forgiv"]),
    ("Whose sword smote and felled whole squadrons of fighting Seraphim?", "pg26-p0150", ["Michael"]),
]
CONDITIONS = [("clean", 0), ("tesseract", 3), ("tesseract", 5)]


def main() -> None:
    rows = []
    for question, gold, must in QUESTIONS:
        print(f"\nQ: {question}  (gold {gold})")
        for source, level in CONDITIONS:
            a = ask(question, source, level)
            retrieved = [p.chunk.page_id for p in a.passages]
            correct = all(m.lower() in a.answer.lower() for m in must)
            row = {"question": question, "gold": gold, "source": source, "level": level,
                   "answer": a.answer, "citations": a.citations, "quote": a.quote,
                   "gold_retrieved": gold in retrieved, "gold_cited": gold in a.citations,
                   "contains_expected": correct, "prompt_tokens": a.prompt_tokens,
                   "completion_tokens": a.completion_tokens}
            rows.append(row)
            print(f"  {source:9s} L{level}  retrieved={'Y' if row['gold_retrieved'] else 'n'} "
                  f"cited={'Y' if row['gold_cited'] else 'n'} correct={'Y' if correct else 'n'}  {a.answer[:110]}")
    Path("results/p3_smoke.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    for source, level in CONDITIONS:
        sel = [r for r in rows if (r["source"], r["level"]) == (source, level)]
        print(f"{source} L{level}: retrieved {sum(r['gold_retrieved'] for r in sel)}/{len(sel)}, "
              f"cited {sum(r['gold_cited'] for r in sel)}/{len(sel)}, "
              f"correct {sum(r['contains_expected'] for r in sel)}/{len(sel)}")


if __name__ == "__main__":
    main()
