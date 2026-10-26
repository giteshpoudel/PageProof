import importlib.util
from pathlib import Path

from pageproof.evals.judge import exact_match


def load(name: str):
    path = Path(__file__).resolve().parents[1] / "evals" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_match():
    assert exact_match("three thousand pounds", "He got Three thousand pounds, instead.")
    assert exact_match("Mrs. Younge", "It was Mrs Younge who helped him.")
    assert not exact_match("three thousand pounds", "He got thirty thousand pounds.")
    assert not exact_match("", "anything")


def test_check():
    gen = load("generate")
    pages = {
        "a": {"page_id": "a", "paragraphs": "Mr. Darcy gave him three thousand pounds in lieu of the living."},
        "b": {"page_id": "b", "paragraphs": "Nothing relevant here at all."},
    }
    good = {"source_page": "a", "question": "How much did Darcy give Wickham instead of the living?",
            "answer": "three thousand pounds", "evidence": "gave him three thousand pounds in lieu of the living"}
    assert gen.check(good, pages) == []
    bad = good | {"evidence": "gave him four thousand pounds"}
    assert "evidence is not an exact quote from the page" in gen.check(bad, pages)
    leaky = good | {"question": "Did Darcy give three thousand pounds?"}
    assert "question contains the answer" in gen.check(leaky, pages)


def test_kappa():
    jv = load("judge_validation")
    assert jv.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(jv.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9
