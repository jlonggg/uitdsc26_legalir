"""Deterministic Vietnamese legal parent/child units for retrieval.

The issued corpus contains noisy HTML-to-text line wrapping, repeated Article
numbers (for example a Decision followed by an attached Regulation), and many
short legal headings.  This module keeps structural instance order in every ID
and never treats an Article number as unique inside a document.
"""
from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Sequence


_STRUCTURE_RE = re.compile(
    r"(?im)^[ \t]*(?P<kind>chương|mục|tiểu[ \t]+mục|điều|phụ(?:[ \t]+|\n[ \t]*)lục)[ \t]+"
    r"(?P<number>[ivxlc]+|\d+[a-zđ]?(?:[.-]\d+)?)\b"
    r"(?P<tail>[^\n]{0,220})"
)
_CLAUSE_RE = re.compile(
    r"(?im)^[ \t]*(?P<label>(?:khoản[ \t]+)?\d{1,3}[a-zđ]?)[ \t]*[.)][ \t]+"
)
_POINT_RE = re.compile(
    r"(?im)^[ \t]*(?P<label>(?:điểm[ \t]+)?[a-zđ])[ \t]*[.)][ \t]+"
)
_BLANK_RE = re.compile(r"\n[ \t]*\n+")
_WS_RE = re.compile(r"\s+")

_BOILERPLATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("scope", re.compile(r"(?i)phạm vi (?:điều chỉnh|áp dụng)|đối tượng áp dụng")),
    ("definition", re.compile(r"(?i)giải thích từ ngữ|trong .*? này,? .*? được hiểu|khái niệm")),
    ("effect", re.compile(r"(?i)hiệu lực (?:thi hành|áp dụng)|có hiệu lực kể từ")),
    ("implementation", re.compile(
        r"(?i)tổ chức thực hiện|trách nhiệm thi hành|chịu trách nhiệm thi hành|điều khoản thi hành"
    )),
)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")


def _compact(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _label(value: str) -> str:
    return _compact(value).casefold()


def _heading(match: re.Match[str]) -> str:
    kind = _compact(match.group("kind")).title()
    number = _compact(match.group("number"))
    tail = _compact(match.group("tail")).lstrip(".:-– ")
    return f"{kind} {number}" + (f". {tail}" if tail else "")


def _is_heading_continuation(line: str) -> bool:
    value = _compact(line)
    if not value or len(value) > 180 or re.search(r"[;?!]", value):
        return False
    letters = [char for char in value if char.isalpha()]
    if not letters:
        return False
    uppercase = sum(char.isupper() for char in letters) / len(letters)
    return uppercase >= 0.72


def _extend_parent_heading(base: str, following: str) -> str:
    """Attach up to three uppercase title lines, without consuming evidence."""
    additions: list[str] = []
    for line in following.splitlines()[:6]:
        if not line.strip():
            if additions:
                break
            continue
        if not _is_heading_continuation(line):
            break
        additions.append(_compact(line))
        if len(additions) == 3:
            break
    return " — ".join([base, *additions])


def _has_substantive_following(following: str) -> bool:
    compact = _compact(following)
    if len(compact) > 240:
        return True
    lines = [_compact(line) for line in following.splitlines() if line.strip()]
    if not lines or len(compact) <= 20:
        return False
    # A short all-uppercase line between Chương/Mục and the next Điều is an
    # inherited title. Mixed-case text or numbered table cells are evidence.
    return not all(_is_heading_continuation(line) for line in lines[:3])


def boilerplate_type(article_heading: str, body: str) -> str | None:
    # Restrict body evidence to the beginning of the unit. A later citation to
    # an effectiveness clause must not turn the current answer unit into one.
    evidence = f"{article_heading}\n{body[:320]}"
    for kind, pattern in _BOILERPLATE_PATTERNS:
        if pattern.search(evidence):
            return kind
    return None


def query_legal_intents(question: str) -> frozenset[str]:
    value = _compact(_norm(question)).casefold()
    intents: set[str] = set()
    if re.search(r"phạm vi|đối tượng áp dụng|áp dụng cho (?:ai|đối tượng)", value):
        intents.add("scope")
    if re.search(r"là gì|được hiểu|khái niệm|định nghĩa|giải thích từ ngữ", value):
        intents.add("definition")
    if re.search(r"hiệu lực|có hiệu lực|áp dụng từ|kể từ ngày nào", value):
        intents.add("effect")
    if re.search(r"tổ chức thực hiện|trách nhiệm thi hành|ai (?:thực hiện|chịu trách nhiệm)", value):
        intents.add("implementation")
    return frozenset(intents)


@dataclass(frozen=True)
class LegalUnit:
    unit_id: str
    doc_id: str
    level: str
    structure_instance: int
    article: str
    clause: str
    point: str
    body: str
    parent_header: str
    boilerplate: str | None
    start: int
    end: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _spans(pattern: re.Pattern[str], text: str) -> list[tuple[re.Match[str], int, int]]:
    matches = list(pattern.finditer(text))
    return [
        (match, match.start(), matches[index + 1].start() if index + 1 < len(matches) else len(text))
        for index, match in enumerate(matches)
    ]


def _paragraph_chunks(text: str, maximum: int = 1400) -> Iterator[tuple[str, int, int]]:
    """Yield deterministic non-overlapping fallbacks for unnumbered Articles."""
    pieces = [(m.start(), m.end()) for m in _BLANK_RE.finditer(text)]
    bounds = [0, *[end for _start, end in pieces], len(text)]
    current = ""
    current_start = 0
    for left, right in zip(bounds, bounds[1:]):
        piece = text[left:right].strip()
        if not piece:
            continue
        if len(piece) > maximum:
            if current:
                yield current, current_start, left
                current = ""
            offset = 0
            while offset < len(piece):
                cut = min(offset + maximum, len(piece))
                if cut < len(piece):
                    space = piece.rfind(" ", offset + maximum // 2, cut + 1)
                    if space > offset:
                        cut = space
                value = piece[offset:cut].strip()
                if value:
                    yield value, left + offset, left + cut
                offset = cut
            continue
        candidate = f"{current}\n{piece}".strip() if current else piece
        if current and len(candidate) > maximum:
            yield current, current_start, left
            current, current_start = piece, left
        else:
            if not current:
                current_start = left
            current = candidate
    if current:
        yield current, current_start, len(text)


def _child_segments(pattern: re.Pattern[str], text: str) -> list[tuple[str, str, int, int]]:
    matches = list(pattern.finditer(text))
    output: list[tuple[str, str, int, int]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        output.append((_label(match.group("label")), text[match.start():end].strip(), match.start(), end))
    return output


def parse_legal_units(doc_id: str, title: str, passage: str) -> list[LegalUnit]:
    """Return Article parents and the smallest safe evidence children.

    `article` records are the Article-level retrieval units (the only level
    corpus.py writes). `clause`, `point`, and `paragraph` records are finer
    atomic units. A point carries the
    short introductory text of its containing Clause because Vietnamese legal
    points frequently depend grammatically on that lead-in.
    """
    text = _norm(passage).strip()
    doc_title = _compact(title) or f"Văn bản {doc_id}"
    structures = list(_STRUCTURE_RE.finditer(text))
    units: list[LegalUnit] = []
    namespace = uuid.UUID("a82a58e1-7ad4-4f9d-8f47-2c0cd56de438")
    chapter = section = ""
    article_instance = 0

    def add(level: str, body: str, article: str, clause: str, point: str,
            parent: str, start: int, end: int, instance: int) -> None:
        value = body.strip()
        if not value:
            return
        uid = str(uuid.uuid5(namespace, f"{doc_id}\n{level}\n{instance}\n{start}\n{end}\n{value}"))
        units.append(LegalUnit(uid, str(doc_id), level, instance, article, clause, point,
                               value, parent, boilerplate_type(article, value), start, end))

    article_regions: list[tuple[int, int, str, str, str, int]] = []
    if structures and structures[0].start() > 0:
        article_regions.append((0, structures[0].start(), "", chapter, section, article_instance))
        article_instance += 1
    elif not structures:
        article_regions.append((0, len(text), "", "", "", article_instance))

    for index, match in enumerate(structures):
        end = structures[index + 1].start() if index + 1 < len(structures) else len(text)
        kind = _compact(match.group("kind")).casefold()
        heading = _heading(match)
        following = text[match.end():end]
        if kind == "chương":
            chapter = _extend_parent_heading(heading, following)
            section = ""
            if _has_substantive_following(following):
                article_regions.append((match.start(), end, heading, chapter, section,
                                        article_instance))
                article_instance += 1
        elif kind in {"mục", "tiểu mục"}:
            section = _extend_parent_heading(heading, following)
            # Standards and technical regulations often use Mục without any
            # Điều. Keep a substantive Mục body as a pseudo-Article parent,
            # while refusing short orphan/header-only candidates.
            if _has_substantive_following(following):
                article_regions.append((match.start(), end, heading, chapter, section,
                                        article_instance))
                article_instance += 1
        elif kind == "phụ lục":
            # An appendix is a real retrieval parent. Without this boundary,
            # the final effectiveness/implementation Article can swallow a
            # hundreds-of-thousands-character table and inherit its type.
            article_regions.append((match.start(), end, heading, chapter, section,
                                    article_instance))
            article_instance += 1
        else:
            article_regions.append((match.start(), end, heading, chapter, section, article_instance))
            article_instance += 1

    for start, end, article, inherited_chapter, inherited_section, instance in article_regions:
        region = text[start:end].strip()
        parent = "\n".join(value for value in
                            (doc_title, inherited_chapter, inherited_section, article) if value)
        # Article is the parent retrieval granularity, but large tables and
        # annexes still need bounded ES records. All parts retain one structural
        # instance and aggregate back to the document by max evidence score.
        for article_part, left, right in _paragraph_chunks(region, maximum=6000):
            add("article", article_part, article, "", "", parent,
                start + left, start + right, instance)

        clauses = _child_segments(_CLAUSE_RE, region)
        if clauses:
            prefix = region[:clauses[0][2]].strip()
            for clause, clause_text, clause_start, clause_end in clauses:
                points = _child_segments(_POINT_RE, clause_text)
                if points:
                    lead = clause_text[:points[0][2]].strip()
                    lead = lead if len(lead) <= 420 else lead[:420].rsplit(" ", 1)[0]
                    for point, point_text, point_start, point_end in points:
                        child_parent = "\n".join(value for value in (parent, clause) if value)
                        for fragment, left, right in _paragraph_chunks(point_text, maximum=1400):
                            body = f"{lead}\n{fragment}".strip()
                            add("point", body, article, clause, point, child_parent,
                                start + clause_start + point_start + left,
                                start + clause_start + point_start + right, instance)
                else:
                    for fragment, left, right in _paragraph_chunks(clause_text, maximum=1400):
                        add("clause", fragment, article, clause, "", parent,
                            start + clause_start + left, start + clause_start + right, instance)
            # The Article title/lead is retrieval context, not a standalone leaf.
            if prefix and len(prefix) > 420:
                for body, left, right in _paragraph_chunks(prefix):
                    add("paragraph", body, article, "", "", parent, start + left, start + right, instance)
        else:
            # A short unnumbered Article is itself the evidence leaf. Long
            # unnumbered material is split only at blank paragraphs/hard cap.
            for body, left, right in _paragraph_chunks(region):
                add("paragraph", body, article, "", "", parent, start + left, start + right, instance)
    return units


def atomic_units(units: Sequence[LegalUnit]) -> Iterable[LegalUnit]:
    return (unit for unit in units if unit.level != "article")
