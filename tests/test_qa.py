import json
import types

from pageproof.index.chunking import MAX_CHARS, Chunk, chunk_page, paragraphs
from pageproof.qa.answer import NOT_FOUND, answer_from, parse_reply
from pageproof.qa.retrieve import SearchResult, rrf


def _chunk(cid: str, page: str = "pg1-p0001") -> Chunk:
    return Chunk(chunk_id=cid, index="t_L0", source="t", level=0, page_id=page, image_id=f"{page}-L0",
                 book_id=1, title="Book", page_no=1, position=0, text=f"text of {cid}")


def test_paragraphs_join_lines_and_rejoin_hyphens():
    assert paragraphs("It is a truth uni-\nversally\nacknowledged\n\nNext para") == [
        "It is a truth universally acknowledged", "Next para"]


def test_chunks_respect_max_size_and_keep_all_words():
    text = "\n\n".join(" ".join(f"w{p}x{i}." for i in range(60)) for p in range(8))
    chunks = chunk_page(text)
    assert all(len(c) <= MAX_CHARS for c in chunks)
    assert " ".join(chunks).split() == " ".join(paragraphs(text)).split()


def test_rrf_rewards_agreement():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    fused = rrf([SearchResult(chunk=a, score=1), SearchResult(chunk=b, score=0.5)],
                [SearchResult(chunk=b, score=9), SearchResult(chunk=c, score=8)])
    assert [r.chunk.chunk_id for r in fused][0] == "b"


def test_parse_reply_tolerates_fences():
    assert parse_reply('```json\n{"answer": "x", "citations": []}\n```')["answer"] == "x"
    assert parse_reply("no json here")["answer"] == NOT_FOUND


def _fake_client(reply: dict):
    def create(**kwargs):
        msg = types.SimpleNamespace(content=json.dumps(reply))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_answer_drops_invented_page_ids():
    results = [SearchResult(chunk=_chunk("a", "pg1-p0001"), score=1)]
    reply = {"answer": "Darcy", "citations": ["pg1-p0001", "pg9-p9999"], "quote": "q", "quote_page": "pg9-p9999"}
    a = answer_from("who?", results, client=_fake_client(reply), model="m")
    assert a.citations == ["pg1-p0001"] and a.quote_page == "pg1-p0001" and a.found


def test_not_found_answer():
    results = [SearchResult(chunk=_chunk("a"), score=1)]
    reply = {"answer": NOT_FOUND, "citations": [], "quote": "", "quote_page": ""}
    assert not answer_from("who?", results, client=_fake_client(reply), model="m").found
