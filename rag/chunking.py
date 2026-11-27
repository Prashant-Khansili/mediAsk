"""Chunking strategies.

``fixed``      – the original 500-character recursive splitter (baseline).
``structured`` – splits on document headings (numbered label sections, highlight
                 banners, Markdown ``#`` headings), keeps tables in one piece,
                 strips page boilerplate and stores drug / section / page metadata.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 50
STRUCTURED_MAX_CHARS = 1200

_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\s+([A-Z][A-Za-z].{1,90})$")
_BANNER_HEADING = re.compile(r"^-*\s*-([A-Z][A-Z ,/&]{3,}[A-Z])\s*-*$")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_TABLE_CAPTION = re.compile(r"^Table\s+\d+[.:]\s+\S")
_DANGLING_WORD = re.compile(r"\b(AND|OR|OF|FOR|THE|WITH|IN|TO)$")
_ONLY_DASHES = re.compile(r"^-{3,}$")
_UUID_PREFIX = re.compile(r"^[0-9a-f]{32}_")
_SENTENCE_END = re.compile(r"(?<=[.;])\s+(?=[A-Z•\[(])")


def fixed_chunks(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_CHUNK_OVERLAP,
    )
    docs = list(docs)
    by_source: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        by_source[doc.metadata.get("source", "")].append(doc)
    drugs = {source: infer_drug_name(pages, source) for source, pages in by_source.items()}

    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        page = chunk.metadata.get("page")
        if isinstance(page, int):
            chunk.metadata["page"] = page + 1  # PyPDFLoader pages are 0-based
        chunk.metadata.setdefault("drug", drugs.get(chunk.metadata.get("source", ""), "Unknown"))
        chunk.metadata.setdefault("section", "")
    return chunks


# --------------------------------------------------------------------------- #
# Structured chunking
# --------------------------------------------------------------------------- #


@dataclass
class _Section:
    heading: str
    page: int
    lines: list[tuple[str, int]] = field(default_factory=list)  # (line, page)


def infer_drug_name(pages: list[Document], source: str) -> str:
    """Best-effort drug name: FDA highlights sentence, Markdown title, filename."""
    head = " ".join(p.page_content for p in pages[:1])
    head = re.sub(r"\s+", " ", head)
    match = re.search(r"needed to use\s+(.+?)\s+safely", head, re.IGNORECASE)
    if match:
        return match.group(1).strip().title()
    match = re.search(r"^#\s+(.+)$", pages[0].page_content if pages else "", re.MULTILINE)
    if match:
        return match.group(1).strip()
    stem = _UUID_PREFIX.sub("", Path(source).stem)
    return stem.replace("_", " ").replace("-", " ").strip().title() or "Unknown"


def _boilerplate_lines(pages: list[Document]) -> set[str]:
    """Lines repeated on most pages (running headers/footers)."""
    if len(pages) < 3:
        return set()
    counts = Counter()
    for page in pages:
        counts.update({line.strip() for line in page.page_content.splitlines() if line.strip()})
    return {line for line, n in counts.items() if n >= max(3, len(pages) // 2)}


def _heading_of(line: str) -> str | None:
    if _ONLY_DASHES.match(line):
        return None
    for pattern in (_MARKDOWN_HEADING, _BANNER_HEADING):
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    match = _NUMBERED_HEADING.match(line)
    if match and not line.rstrip().endswith((".", ",", ";")):
        return f"{match.group(1)} {match.group(2).strip()}"
    if line.startswith("WARNING:") and line.upper() == line:
        return line
    words = line.split()
    if (
        len(words) >= 3
        and len(line) <= 60
        and line.upper() == line
        and re.search(r"[A-Z]{3}", line)
        and not re.search(r"\d", line)
    ):
        return line.rstrip(":*")
    return None


def _is_prose(line: str) -> bool:
    lowercase_words = re.findall(r"\b[a-z]{2,}\b", line)
    numbers = re.findall(r"\d+(?:[.,]\d+)?", line)
    return len(line) >= 70 and len(lowercase_words) >= 8 and len(numbers) <= 1


def _units(lines: list[tuple[str, int]]) -> list[tuple[str, int, bool]]:
    """Group section lines into (text, page, is_table) units.

    Tables (caption through the last tabular row, plus any ``*`` footnote) form a
    single unit; prose is split into sentences.
    """
    units: list[tuple[str, int, bool]] = []
    prose: list[str] = []
    prose_page = None
    index = 0

    def flush_prose():
        nonlocal prose, prose_page
        if prose:
            text = " ".join(prose)
            for sentence in _SENTENCE_END.split(text):
                if sentence.strip():
                    units.append((sentence.strip(), prose_page, False))
        prose, prose_page = [], None

    while index < len(lines):
        line, page = lines[index]
        if _TABLE_CAPTION.match(line):
            flush_prose()
            table = [line]
            index += 1
            if index < len(lines):  # caption continuation / column header
                table.append(lines[index][0])
                index += 1
            in_footnote = False
            while index < len(lines):
                row = lines[index][0]
                if in_footnote:
                    table.append(row)
                    index += 1
                    if row.rstrip().endswith("."):
                        break
                    continue
                if row.startswith("*"):
                    in_footnote = True
                    table.append(row)
                    index += 1
                    if row.rstrip().endswith("."):
                        break
                    continue
                if _TABLE_CAPTION.match(row) or _is_prose(row):
                    break
                table.append(row)
                index += 1
            units.append(("\n".join(table), page, True))
            continue
        if prose_page is None:
            prose_page = page
        prose.append(line)
        index += 1
    flush_prose()
    return units


def _pack(units: list[tuple[str, int, bool]], max_chars: int) -> list[tuple[str, int, int]]:
    """Greedily pack units into chunks <= max_chars; tables are never split."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=100)
    expanded: list[tuple[str, int, bool]] = []
    for text, page, is_table in units:
        if not is_table and len(text) > max_chars:  # run-on prose with no sentence breaks
            expanded.extend((piece, page, False) for piece in splitter.split_text(text))
        else:
            expanded.append((text, page, is_table))
    units = expanded
    chunks: list[tuple[str, int, int]] = []
    current: list[str] = []
    first_page = last_page = None
    size = 0
    for text, page, _is_table in units:
        if current and size + len(text) + 1 > max_chars:
            chunks.append(("\n".join(current), first_page, last_page))
            # one-sentence overlap for prose continuity
            tail = current[-1] if len(current[-1]) < 300 and "\n" not in current[-1] else None
            current = [tail] if tail else []
            size = len(tail) if tail else 0
            first_page = last_page if tail else None
        current.append(text)
        size += len(text) + 1
        first_page = page if first_page is None else first_page
        last_page = page
    if current:
        chunks.append(("\n".join(current), first_page, last_page))
    return chunks


def structured_chunks(docs: Iterable[Document], max_chars: int = STRUCTURED_MAX_CHARS) -> list[Document]:
    by_source: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        by_source[doc.metadata.get("source", "")].append(doc)

    chunks: list[Document] = []
    for source, pages in by_source.items():
        pages = sorted(pages, key=lambda d: d.metadata.get("page", 0))
        drug = infer_drug_name(pages, source)
        boilerplate = _boilerplate_lines(pages)

        sections: list[_Section] = [_Section(heading="", page=1)]
        for page_doc in pages:
            page_number = int(page_doc.metadata.get("page", 0)) + 1
            in_contents = False
            for raw in page_doc.page_content.splitlines():
                line = raw.strip()
                if not line or line in boilerplate or _ONLY_DASHES.match(line):
                    continue
                if in_contents:  # table-of-contents page: nothing to retrieve
                    continue
                current = sections[-1]
                if not current.lines and _DANGLING_WORD.search(current.heading):
                    current.heading = f"{current.heading} {line}"  # heading wrapped onto two lines
                    continue
                heading = _heading_of(line)
                if heading and "CONTENTS" in heading.upper():
                    in_contents = True
                    continue
                if heading:
                    sections.append(_Section(heading=heading, page=page_number))
                else:
                    current.lines.append((line, page_number))

        for section in sections:
            if not section.lines:
                continue
            for text, first, last in _pack(_units(section.lines), max_chars):
                header = f"[{drug}" + (f" | {section.heading}" if section.heading else "") + "]"
                chunks.append(
                    Document(
                        page_content=f"{header}\n{text}",
                        metadata={
                            "source": source,
                            "drug": drug,
                            "section": section.heading,
                            "page": first,
                            "page_end": last,
                        },
                    )
                )
    return chunks


STRATEGIES = {"fixed": fixed_chunks, "structured": structured_chunks}


def chunk_documents(docs: Iterable[Document], strategy: str = "structured") -> list[Document]:
    try:
        chunker = STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose one of: {', '.join(STRATEGIES)}") from exc
    chunks = chunker(docs)
    for number, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = number
    return chunks
