"""Corpus text normalisation, tokenisation and chunking helpers; no model or artifact readers."""

from __future__ import annotations

import hashlib

import json

import os

import platform

import random

import re

import time

import unicodedata

import uuid

import zipfile

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Any, Iterator, Mapping, Sequence

class ConfigurationError(RuntimeError):
    """Raised when configuration or a competition invariant is violated."""

@dataclass(frozen=True)
class LegalDocument:
    doc_id: str
    name: str
    link: str
    passage: str

@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    doc_id: str
    name: str
    link: str
    article: str
    order: int
    text: str
    normalized: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LegalChunk":
        return cls(
            chunk_id=str(value["chunk_id"]),
            doc_id=str(value["doc_id"]),
            name=str(value.get("name", "")),
            link=str(value.get("link", "")),
            article=str(value.get("article", "")),
            order=int(value.get("order", 0)),
            text=str(value["text"]),
            normalized=str(value.get("normalized") or normalize_text(str(value["text"]))),
        )

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).replace("đ", "đ").replace("Đ", "đ")
    return re.sub(r"\s+", " ", text).strip().lower()

_LEGAL_TOKEN_RE = re.compile(
    r"\d+(?:[./-]\d+)+(?:[/-][a-zA-ZĐđ0-9]+)*|[a-zA-ZÀ-ỹĐđ]+(?:[-'][a-zA-ZÀ-ỹĐđ]+)*|\d+",
    re.UNICODE,
)

def legal_tokenize(text: str) -> list[str]:
    """Tokenize consistently while preserving legal document numbers."""
    return _LEGAL_TOKEN_RE.findall(normalize_text(text))

def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "d"))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

def title_tokens(text: str) -> list[str]:
    """Tokens for matching a query against a document title.

    Document names are unaccented URL slugs ("...Dieu-le-Hiep-hoi-Dieu-Viet-Nam-208881"),
    but `_LEGAL_TOKEN_RE` keeps a hyphen chain as ONE token and `normalize_text` never
    strips accents, so an accented question almost never shares a `legal_tokenize` token
    with a slug. Splitting on hyphens and folding accents makes the two sides comparable.
    Apply it to BOTH index and query text, or the halves stop matching.
    """
    return [tok for tok in re.split(r"[^a-z0-9]+", _strip_accents(normalize_text(text))) if tok]

def lexical_text(text: str) -> str:
    return " ".join(legal_tokenize(text))

_SLUG_TAIL_RE = re.compile(r"-\d{4,}$")

def readable_title(name: str) -> str:
    """Turn a URL slug into something the embedder can read.

    "Quyet-dinh-948-QD-BNV-nam-2013-Dieu-le-Hiep-hoi-Dieu-Viet-Nam-208881"
    -> "Quyet dinh 948 QD BNV nam 2013 Dieu le Hiep hoi Dieu Viet Nam".
    The trailing corpus id is dropped: it is a thuvienphapluat page number, never
    something a question refers to, and it only adds a junk token to the vector.
    """
    return _SLUG_TAIL_RE.sub("", str(name)).replace("-", " ").strip()

def dense_text(chunk: "LegalChunk") -> str:
    """Text actually embedded for a chunk.

    Document title + article header + body. Embedding the bare body alone would make
    structurally identical documents (e.g. two decisions with the same template)
    indistinguishable; BM25 and the reranker also see the document name.
    """
    parts = [readable_title(chunk.name), chunk.article, chunk.text]
    return normalize_text(" ".join(part for part in parts if part))

def title_text(text: str) -> str:
    return " ".join(title_tokens(text))

_BOUNDARY_RE = re.compile(r"(?im)(?=^[ \t]*(?:chương|mục|tiểu mục|điều)\s+[\divxlc]+\b[^\n]*)")

_ARTICLE_RE = re.compile(r"(?im)^\s*(điều\s+\d+[a-zđ]?(?:[.-]\d+)?)\b")

_CLAUSE_RE = re.compile(r"(?m)(?=^\s*(?:\d+[.)]|[a-zđ][.)]|[-–•])\s+)")

_HIERARCHY_LINE_RE = re.compile(r"(?im)^\s*((?:chương|mục|tiểu mục|điều)\s+[\divxlc]+\b[^\n]*)")

def _hard_split(text: str, maximum: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > maximum:
        cut = remaining.rfind(" ", 0, maximum + 1)
        if cut < maximum // 2:
            cut = maximum
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts

def _pack_units(units: Sequence[str], target: int, maximum: int) -> list[str]:
    expanded = [piece for unit in units for piece in _hard_split(unit, maximum) if piece]
    packed: list[str] = []
    current = ""
    for unit in expanded:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if current and len(candidate) > maximum:
            packed.append(current)
            current = unit
        else:
            current = candidate
        if len(current) >= target:
            packed.append(current)
            current = ""
    if current:
        if packed and len(packed[-1]) + len(current) + 1 <= maximum:
            packed[-1] = f"{packed[-1]}\n{current}"
        else:
            packed.append(current)
    return packed

def chunk_document(
    doc_id: str,
    name: str,
    link: str,
    passage: str,
    target_chars: int = 1500,
    max_chars: int = 1800,
) -> list[LegalChunk]:
    if target_chars <= 0 or max_chars < target_chars:
        raise ConfigurationError("chunk target/max must satisfy 0 < target <= max")
    normalized_newlines = unicodedata.normalize("NFKC", passage).replace("\r\n", "\n").replace("\r", "\n").strip()
    sections = [section.strip() for section in _BOUNDARY_RE.split(normalized_newlines) if section.strip()]
    if not sections:
        sections = [normalized_newlines]
    chunk_texts: list[tuple[str, str]] = []
    current_article = ""
    for section in sections:
        match = _ARTICLE_RE.search(section)
        if match:
            current_article = match.group(1).strip()
        units = [unit.strip() for unit in _CLAUSE_RE.split(section) if unit.strip()]
        if len(units) == 1:
            units = [unit.strip() for unit in re.split(r"\n\s*\n+", section) if unit.strip()]
        for packed in _pack_units(units, target_chars, max_chars):
            chunk_texts.append((current_article, packed))
    result: list[LegalChunk] = []
    namespace = uuid.UUID("7f1123cb-585d-4f7a-bab5-39a90c428314")
    for order, (article, text) in enumerate(chunk_texts):
        chunk_id = str(uuid.uuid5(namespace, f"{doc_id}\n{order}\n{text}"))
        result.append(LegalChunk(chunk_id, doc_id, name, link, article, order, text, normalize_text(text)))
    return result

def _link_title(link: str) -> str:
    """A deterministic title fallback; issued contexts often omit ``name``."""
    tail = link.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"[-_]+", " ", tail).strip()

def chunk_document_hierarchical(
    doc_id: str,
    name: str,
    link: str,
    passage: str,
    target_chars: int = 1200,
    max_chars: int = 1400,
) -> list[LegalChunk]:
    """Chunk Article bodies while attaching their inherited legal path.

    Unlike `chunk_document`, structural headings are never emitted as standalone
    chunks. A clause is encoded with the document/chapter/section/article context
    that gives it legal meaning. Selected by `chunking.hierarchical_context`;
    corpus.py does not enable it.
    """
    if target_chars <= 0 or max_chars < target_chars:
        raise ConfigurationError("chunk target/max must satisfy 0 < target <= max")
    text = unicodedata.normalize("NFKC", passage).replace("\r\n", "\n").replace("\r", "\n").strip()
    document_title = name.strip() or _link_title(link) or f"document {doc_id}"
    chapter = section = article = ""
    article_heading = ""
    body: list[str] = []
    raw_units: list[tuple[str, str, str, str, str]] = []

    def flush() -> None:
        nonlocal chapter, section
        if not body:
            return
        value = "\n".join(body).strip()
        # Many issued documents put a short title on the line after ``Chương`` or
        # ``Mục``. It is context, not evidence: attach it to the parent instead
        # of manufacturing a header-only candidate.
        if value and not article and len(value) <= 240 and not re.search(r"[.;:?!]", value):
            compact_value = re.sub(r"\s+", " ", value)
            if section:
                section = f"{section} — {compact_value}"
            elif chapter:
                chapter = f"{chapter} — {compact_value}"
            body.clear()
            return
        if value:
            raw_units.append((chapter, section, article, article_heading, value))
        body.clear()

    cursor = 0
    for match in _HIERARCHY_LINE_RE.finditer(text):
        before = text[cursor:match.start()].strip()
        if before:
            body.append(before)
        heading = re.sub(r"\s+", " ", match.group(1)).strip()
        kind = heading.split(None, 1)[0].casefold()
        if kind == "chương":
            flush()
            chapter, section, article, article_heading = heading, "", "", ""
        elif kind in {"mục", "tiểu"}:
            flush()
            section, article, article_heading = heading, "", ""
        else:  # Điều
            flush()
            article, article_heading = heading, heading
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        body.append(tail)
    flush()
    if not raw_units:
        raw_units = [("", "", "", "", text)]

    result: list[LegalChunk] = []
    namespace = uuid.UUID("bb39c87f-4740-46cc-923a-c6f5d8692d0f")
    for inherited_chapter, inherited_section, inherited_article, heading, unit in raw_units:
        prefix = "\n".join(
            value for value in (document_title, inherited_chapter, inherited_section, heading) if value
        )
        body_max = max(200, max_chars - len(prefix) - 1)
        body_target = min(target_chars, body_max)
        clause_units = [value.strip() for value in _CLAUSE_RE.split(unit) if value.strip()] or [unit]
        for packed in _pack_units(clause_units, body_target, body_max):
            contextualized = f"{prefix}\n{packed}".strip()
            order = len(result)
            chunk_id = str(uuid.uuid5(namespace, f"{doc_id}\n{order}\n{contextualized}"))
            result.append(LegalChunk(chunk_id, doc_id, name, link, inherited_article, order, contextualized, normalize_text(contextualized)))
    return result

def build_chunks(documents: Sequence[LegalDocument], config: Mapping[str, Any]) -> list[LegalChunk]:
    settings = config.get("chunking", {})
    target = int(settings.get("target_chars", 1500))
    maximum = int(settings.get("max_chars", 1800))
    hierarchical = bool(settings.get("hierarchical_context", False))
    chunker = chunk_document_hierarchical if hierarchical else chunk_document
    return [chunk for document in documents for chunk in chunker(document.doc_id, document.name, document.link, document.passage, target, maximum)]
