#!/usr/bin/env python3
"""Build an auditable legal-reference graph from the issued DSC26 contexts.

The input corpus has no canonical legal-document metadata beyond ``id``, ``link``
and ``passage``.  This extractor therefore resolves a cross-document reference
only when its document number maps to exactly one corpus context.  Ambiguous and
unresolved citations remain in the graph, but are deliberately not expandable.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


ARTICLE = re.compile(r"(?im)^\s*(điều\s+(\d+[a-zđ]?)(?:\s*[.:]\s*[^\n]{0,160})?)")
CLAUSE = re.compile(r"(?im)^\s*(khoản\s+(\d+[a-zđ]?)(?:\s*[.:]\s*[^\n]{0,120})?)")
DOC_NUMBER_CORE = (
    r"\d{1,6}\s*(?:/|-)\s*"
    r"(?:(?:\d{2,4})\s*(?:/|-)\s*)?"
    r"[a-zđ][a-zđ0-9]{0,19}(?:\s*-\s*[a-zđ0-9]{1,20})*"
)
DOC_NO = re.compile(rf"(?i)(?:số\s*:?\s*)?({DOC_NUMBER_CORE})")
HEADER_DOC_NO = re.compile(
    rf"(?im)^\s*(?:(?:bộ\s+)?luật\s+|nghị\s+quyết\s+)?số\s*:\s*({DOC_NUMBER_CORE})"
)
INTERNAL_ARTICLE = re.compile(r"(?i)điều\s+(\d+[a-zđ]?)\s+(?:của\s+)?(?:luật|nghị định|thông tư|quyết định|văn bản|bộ luật)\s+này")
EXTERNAL_ARTICLE = re.compile(
    r"(?is)điều\s+(\d+[a-zđ]?)(?:\s*,?\s*khoản\s+(\d+[a-zđ]?))?"
    r".{0,160}?(?:của|tại)\s+(?:bộ luật|luật|nghị định|thông tư|quyết định|"
    r"pháp lệnh|công văn)[^.;\n]{0,220}?"
    rf"(?:số\s*:?\s*)?({DOC_NUMBER_CORE})"
)
BARE_DOC = re.compile(
    r"(?is)(?:bộ luật|luật|nghị định|thông tư|quyết định|pháp lệnh|công văn)"
    rf"[^.;\n]{{0,220}}?(?:số\s*:?\s*)?({DOC_NUMBER_CORE})"
)


def norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")


def norm_no(value: str) -> str:
    return re.sub(r"[^a-z0-9đ]", "", value.lower())


def document_identity(text: str) -> dict[str, Any]:
    """Return only a canonical number stated in the document header.

    Legal texts cite many other instruments in their recitals.  Those numbers
    are references, not aliases for the current document, and must never enter
    the canonical resolver index.
    """
    header = text[:1500]
    match = HEADER_DOC_NO.search(header)
    canonical_raw = re.sub(r"\s+", "", match.group(1)) if match else None
    canonical = norm_no(canonical_raw) if canonical_raw else None
    referenced = sorted({
        norm_no(item.group(1))
        for item in DOC_NO.finditer(text[:12000])
        if norm_no(item.group(1)) != canonical
    })
    return {
        "canonical_document_number": canonical_raw,
        "canonical_document_number_normalized": canonical,
        "canonical_resolution_source": "header" if canonical else None,
        # Kept for consumers that read `document_numbers`; it holds only the
        # canonical identity, never recital/body citations.
        "document_numbers": [canonical] if canonical else [],
        "referenced_document_numbers": referenced,
    }


def excerpts(text: str, start: int, end: int, limit: int = 500) -> str:
    lo = max(0, text.rfind("\n", 0, start - 1) + 1)
    hi = text.find("\n", end)
    if hi < 0:
        hi = len(text)
    value = re.sub(r"\s+", " ", text[lo:hi]).strip()
    return value[:limit]


def relation_for(text: str, pos: int) -> str:
    window = text[max(0, pos - 180): pos + 180].lower()
    if re.search(r"sửa đổi|bổ sung", window):
        return "amends"
    if re.search(r"thay thế|bãi bỏ", window):
        return "replaces_or_repeals"
    if re.search(r"hướng dẫn thi hành|quy định chi tiết", window):
        return "implements"
    if re.search(r"căn cứ", window):
        return "legal_basis"
    return "cross_reference"


def source_node(text: str, pos: int) -> tuple[str | None, str | None]:
    articles = list(ARTICLE.finditer(text, 0, pos + 1))
    clauses = list(CLAUSE.finditer(text, 0, pos + 1))
    article = articles[-1].group(2).lower() if articles else None
    # A clause belongs to the current article only if it occurs after that article.
    clause = clauses[-1].group(2).lower() if clauses and (not articles or clauses[-1].start() >= articles[-1].start()) else None
    return article, clause


def iter_contexts(context_dir: Path, paths: list[Path] | None = None) -> Iterator[tuple[str, str, str]]:
    for path in (paths if paths is not None else sorted(context_dir.glob("context_*.json"))):
        try:
            row = json.loads(path.read_text(encoding="utf-8-sig"))
            yield str(row["id"]), str(row.get("link") or ""), norm(str(row.get("passage") or ""))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(f"cannot read {path}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, default=Path(__file__).parent / "selected-contexts")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--stage", choices=("extract", "resolve", "all"), default="all")
    parser.add_argument("--start", type=int, default=0, help="zero-based context-file offset for a restartable extract batch")
    parser.add_argument("--count", type=int, default=0, help="number of files in an extract batch; 0 means all remaining")
    parser.add_argument("--append", action="store_true", help="append an extract batch to existing intermediate files")
    args = parser.parse_args()
    files = list(sorted(args.contexts.glob("context_*.json")))
    if not files:
        raise SystemExit(f"no context_*.json in {args.contexts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ``extract`` makes exactly one pass over the 483MB corpus.  Keeping resolve
    # separate permits a short, restartable job on network-backed storage.
    number_to_docs: dict[str, set[str]] = defaultdict(set)
    doc_meta: dict[str, dict[str, Any]] = {}
    graph_path = args.output_dir / "legal_reference_graph.jsonl"
    nodes_path = args.output_dir / "legal_reference_nodes.jsonl"
    stats_path = args.output_dir / "legal_reference_graph_stats.json"
    pending_path = args.output_dir / "legal_reference_pending.jsonl"
    catalog_path = args.output_dir / "legal_reference_document_catalog.jsonl"

    if args.stage in {"extract", "all"}:
        extract_stats: Counter[str] = Counter()
        selected = files[args.start: args.start + args.count if args.count else None]
        mode = "a" if args.append else "w"
        with nodes_path.open(mode, encoding="utf-8") as nodes, pending_path.open(mode, encoding="utf-8") as pending, catalog_path.open(mode, encoding="utf-8") as catalog_out:
            for source_doc, link, text in iter_contexts(args.contexts, selected):
                identity = document_identity(text)
                numbers = set(identity["document_numbers"])
                doc_meta[source_doc] = {"doc_id": source_doc, "link": link, **identity}
                catalog_out.write(json.dumps(doc_meta[source_doc], ensure_ascii=False) + "\n")
                for number in numbers:
                    number_to_docs[number].add(source_doc)
                articles = list(ARTICLE.finditer(text))
                for i, match in enumerate(articles):
                    end = articles[i + 1].start() if i + 1 < len(articles) else len(text)
                    nodes.write(json.dumps({"node_type": "article", "node_id": f"{source_doc}:article:{match.group(2).lower()}", "doc_id": source_doc, "article": match.group(2).lower(), "heading": re.sub(r"\s+", " ", match.group(1)).strip()[:300], "start": match.start(), "end": end}, ensure_ascii=False) + "\n")
                    extract_stats["article_nodes"] += 1
                candidates: list[tuple[re.Match[str], str, str | None, str | None, str]] = []
                for match in INTERNAL_ARTICLE.finditer(text):
                    candidates.append((match, "same_document_reference", match.group(1), None, ""))
                for match in EXTERNAL_ARTICLE.finditer(text):
                    candidates.append((match, relation_for(text, match.start()), match.group(1), match.group(2), norm_no(match.group(3))))
                for match in BARE_DOC.finditer(text):
                    candidates.append((match, relation_for(text, match.start()), None, None, norm_no(match.group(1))))
                for match, relation, target_article, target_clause, doc_number in candidates:
                    src_article, src_clause = source_node(text, match.start())
                    pending.write(json.dumps({"source_doc_id": source_doc, "source_article": src_article, "source_clause": src_clause, "relation": relation, "raw_reference": re.sub(r"\s+", " ", match.group(0)).strip()[:700], "target_article": target_article.lower() if target_article else None, "target_clause": target_clause.lower() if target_clause else None, "document_number_normalized": doc_number or None, "evidence": excerpts(text, match.start(), match.end())}, ensure_ascii=False) + "\n")
                    extract_stats["pending_edges"] += 1
                extract_stats["contexts"] += 1
            for doc in doc_meta.values():
                nodes.write(json.dumps({"node_type": "document", **doc}, ensure_ascii=False) + "\n")
        if args.stage == "extract":
            print(json.dumps(dict(extract_stats), ensure_ascii=False))
            return

    if not catalog_path.exists() or not pending_path.exists():
        raise SystemExit("run --stage extract before --stage resolve")
    catalog_docs = [json.loads(line) for line in catalog_path.open(encoding="utf-8") if line.strip()]
    for doc in catalog_docs:
        for number in doc["document_numbers"]:
            number_to_docs[number].add(doc["doc_id"])
    stats: Counter[str] = Counter(contexts=len({doc["doc_id"] for doc in catalog_docs}))
    seen: set[tuple[Any, ...]] = set()
    with graph_path.open("w", encoding="utf-8") as out, pending_path.open(encoding="utf-8") as pending:
        for raw in pending:
            item = json.loads(raw)
            source_doc, relation = item["source_doc_id"], item["relation"]
            target_docs = {source_doc} if relation == "same_document_reference" else number_to_docs.get(item["document_number_normalized"] or "", set())
            # Drop exact duplicates (same source node, relation, targets and raw reference text).
            key = (source_doc, item["source_article"], item["source_clause"], relation, tuple(sorted(target_docs)), item["target_article"], item["target_clause"], item["raw_reference"])
            if key in seen:
                continue
            seen.add(key)
            status = "resolved" if len(target_docs) == 1 else ("ambiguous" if target_docs else "unresolved")
            edge = {
                **item,
                "target_doc_id": next(iter(target_docs)) if len(target_docs) == 1 else None,
                "resolution": status,
                "expandable": status == "resolved" and relation in {"cross_reference", "same_document_reference", "amends", "replaces_or_repeals", "implements"},
            }
            out.write(json.dumps(edge, ensure_ascii=False) + "\n")
            stats["edges"] += 1
            stats[f"relation:{relation}"] += 1
            stats[f"resolution:{status}"] += 1
            if edge["expandable"]:
                stats["expandable_edges"] += 1

    stats["unique_document_numbers"] = len(number_to_docs)
    stats["unique_resolvable_document_numbers"] = sum(len(v) == 1 for v in number_to_docs.values())
    result: dict[str, Any] = dict(sorted(stats.items()))
    result["schema_version"] = 2
    result["resolver_policy"] = "Only exact, unique canonical document numbers extracted from a document header are expandable; recital/body citations never enter the identity index."
    stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
