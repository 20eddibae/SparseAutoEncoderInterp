"""
ShareGPT loader. Expects a local JSON file in the format
[{"conversations": [{"from": "human"|"gpt", "value": "..."}, ...]}, ...].

We don't auto-download — the file is ~700MB. See docs/HPC_PLAYBOOK.md for
where to fetch it.
"""
from __future__ import annotations
import html
import json
import os
import re
from html.parser import HTMLParser
from typing import Iterable

from ..config import Config
from .types import Conversation, Turn

_ROLE_MAP = {"human": "human", "user": "human", "gpt": "ai", "assistant": "ai"}

# Block-level tags whose boundaries should become whitespace so stripping HTML
# doesn't mash adjacent words/lines together (e.g. </p><p> -> a line break).
_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "pre", "tr", "table",
               "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "hr"}
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    """Stdlib-only HTML -> text. Drops tags, keeps text, inserts newlines at
    block boundaries, skips <script>/<style> contents."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def strip_html(text: str) -> str:
    """Render HTML markup to plain text. Idempotent on already-plain text."""
    if "<" not in text and "&" not in text:
        return text
    p = _TextExtractor()
    try:
        p.feed(text)
        p.close()
        out = p.text()
    except Exception:
        # Malformed markup: fall back to a crude tag strip rather than dropping the turn.
        out = re.sub(r"<[^>]+>", " ", text)
    out = html.unescape(out)
    out = _WS_RE.sub(" ", out)
    out = _NL_RE.sub("\n\n", out)
    return out.strip()


def iter_sharegpt(cfg: Config) -> Iterable[Conversation]:
    path = cfg.corpus.sharegpt_path
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ShareGPT file not found at {path}. See docs/HPC_PLAYBOOK.md."
        )
    with open(path) as f:
        data = json.load(f)

    yielded = 0
    for i, item in enumerate(data):
        raw_turns = item.get("conversations") or item.get("messages") or []
        turns: list[Turn] = []
        for t in raw_turns[: cfg.corpus.max_turns]:
            role_in = (t.get("from") or t.get("role") or "").lower()
            role = _ROLE_MAP.get(role_in)
            text = t.get("value") or t.get("content") or ""
            if cfg.corpus.strip_html:
                text = strip_html(text)
            if role is None or not text.strip():
                continue
            turns.append(Turn(role=role, text=text))
        if len(turns) < 2:
            continue
        yield Conversation(turns=turns, meta={"source": "sharegpt", "idx": i})
        yielded += 1
        if yielded >= cfg.corpus.n_conversations:
            break
