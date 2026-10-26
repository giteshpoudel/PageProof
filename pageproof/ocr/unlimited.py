"""Unlimited-OCR (baidu/Unlimited-OCR) through Hugging Face Transformers.

The released model code assumes CUDA. To run it on Apple Silicon (MPS) or CPU we
patch a local snapshot, following community PR baidu/Unlimited-OCR#56:

1. Image-embedding injection: ``masked_scatter_`` with a broadcast mask
   mis-scatters on MPS (issue #18), giving empty output. Replace it with
   explicit positional assignment, which is identical on every device.
2. Every ``.cuda()`` follows the model's device instead.
3. ``torch.autocast`` runs only on CUDA: bf16 autocast on MPS makes logits drift
   into coordinate loops (reported in PR #57).

The weights stay in bfloat16 (~6.7 GB), so the model fits a 16 GB Mac.
On CUDA the patched code behaves exactly like the original.
"""

import os
import re
from pathlib import Path

PROMPT = "<image>document parsing."
# "gundam" preset from the model README: tiles the page, best for dense text
GUNDAM = {"base_size": 1024, "image_size": 640, "crop_mode": True}

_INJECTION_BEFORE = (
    "                    inputs_embeds[idx].masked_scatter_("
    "images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)"
)
_INJECTION_AFTER = """\
                    # PageProof patch (after PR #56): masked_scatter_ mis-scatters
                    # on MPS with a broadcast mask; positional assignment is identical.
                    _feat = images_in_this_batch.to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )
                    _pos = (
                        images_seq_mask[idx]
                        .to(inputs_embeds.device)
                        .bool()
                        .nonzero(as_tuple=True)[0]
                    )
                    inputs_embeds[idx, _pos] = _feat"""

# (pattern, replacement, expected count in the released file)
_DEVICE_EDITS = [
    (".unsqueeze(0).cuda().shape[1]", ".unsqueeze(0).shape[1]", 4),
    ("input_ids.unsqueeze(0).cuda()", "input_ids.unsqueeze(0).to(self.device)", 3),
    ("images_seq_mask.unsqueeze(0).cuda()", "images_seq_mask.unsqueeze(0).to(self.device)", 3),
    ("images_crop.cuda()", "images_crop.to(self.device)", 2),
    ("images_ori.cuda()", "images_ori.to(self.device)", 3),
    ("dummy_crop.cuda()", "dummy_crop.to(self.device)", 1),
    ('with torch.autocast("cuda", dtype=torch.bfloat16):',
     'with torch.autocast(self.device.type, dtype=torch.bfloat16, '
     'enabled=(self.device.type == "cuda")):', 3),
]


def patch_snapshot(model_dir: Path) -> bool:
    """Patch modeling_unlimitedocr.py in place. Returns False if already patched.

    Asserts on the exact released code, so it fails loudly if upstream changes.
    """
    path = model_dir / "modeling_unlimitedocr.py"
    src = path.read_text(encoding="utf-8")
    if "inputs_embeds[idx, _pos] = _feat" in src and ".cuda()" not in src:
        return False
    assert _INJECTION_BEFORE in src, "released injection line not found; upstream changed"
    src = src.replace(_INJECTION_BEFORE, _INJECTION_AFTER)
    for pattern, replacement, expected in _DEVICE_EDITS:
        found = src.count(pattern)
        assert found == expected, f"expected {expected}x {pattern!r}, found {found}"
        src = src.replace(pattern, replacement)
    assert ".cuda()" not in src, "an unpatched .cuda() call remains"
    path.write_text(src, encoding="utf-8")
    return True


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class UnlimitedOCR:
    """Single-page inference. Multi-page parsing degrades on MPS (issue #53)."""

    def __init__(self, model_dir: str | Path, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_dir = Path(model_dir)
        self.device = device or pick_device()
        if self.device != "cuda":
            patch_snapshot(self.model_dir)
        if self.device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        self.model = model.eval().to(self.device)

    def ocr_page(self, image_path: str, out_dir: str = "/tmp/pageproof-uocr",
                 max_length: int = 8192) -> str:
        """Raw model output for one page (Markdown with <|det|> layout tags)."""
        import torch

        with torch.no_grad():
            return self.model.infer(
                self.tokenizer,
                prompt=PROMPT,
                image_file=image_path,
                output_path=out_dir,
                **GUNDAM,
                max_length=max_length,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=False,
                eval_mode=True,
            )


# A layout header: an optional region label (<|ref|>text<|/ref|>, the vLLM format)
# followed by its box (<|det|>...<|/det|>). Both go; the page text follows it.
_DET = re.compile(r"(<\|ref\|>[^<]*<\|/ref\|>\s*)?<\|det\|>.*?<\|/det\|>", re.S)
_TAG = re.compile(r"<\|[^|]*\|>")
_MD = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|>\s+)", re.M)


_LATEX_INLINE = re.compile(r"\\\((.*?)\\\)", re.S)
_LATEX_SCRIPT = re.compile(r"[\^_]\{([^{}]*)\}")


def _flatten_latex(math: str) -> str:
    # The model writes superscripts such as "Mr." as \( M^{r} \); keep the letters.
    return _LATEX_SCRIPT.sub(r"\1", math).replace(" ", "")


def to_plain_text(raw: str) -> str:
    """Strip layout tags, inline LaTeX and Markdown markup so output compares with plain ground truth."""
    text = _TAG.sub("", _DET.sub("", raw))
    text = _LATEX_INLINE.sub(lambda m: _flatten_latex(m.group(1)), text)
    text = re.sub(r"(\w) \.", r"\1.", text)  # "M^{rs} ." -> "Mrs."
    text = _MD.sub("", text)
    text = re.sub(r"(\*\*|__|\*|_)(\S.*?\S|\S)\1", r"\2", text)  # bold / italic
    return text
