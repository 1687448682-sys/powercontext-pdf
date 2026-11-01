# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Complete PDF capture and offline retrieval of exact physical-page evidence.

Only the parser consumes PDF bytes. Retrieval uses the caller's scoped Memory;
neither the original files nor an auxiliary index are needed after ingestion.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pypdf import PdfReader, get_configuration
from pypdf.filters import decode_stream_data
from pypdf.generic import (
    ArrayObject,
    ContentStream,
    DictionaryObject,
    NameObject,
    NullObject,
    NumberObject,
    PdfObject,
    StreamObject,
)

from powercontext.artifacts import MemoryCitation
from powercontext.sources import AdapterSourceDefinition, Source, SourceMaterialization, TextEvidence
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import ScopedMemoryApplication
    from powercontext.builtin.runtime.models import MemoryEntryRecord


class PDFDocumentInput(BaseModel):
    """An entire original PDF, identified by the caller's document identifier."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    """One physical, one-based page from an exact PDF byte revision."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(ge=1, strict=True)
    text: str

    @field_validator("document_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")  # noqa: TRY003
        return value


def _page_tree_count(value: PdfObject, seen: set[int]) -> int:
    node = value.get_object()
    if not isinstance(node, DictionaryObject) or id(node) in seen:
        raise ValueError("Invalid or repeated page-tree node")  # noqa: TRY003
    seen.add(id(node))
    if node.get("/Type") == "/Page":
        return 1
    kids, declared = node.get("/Kids"), node.get("/Count")
    kids = kids.get_object() if kids is not None else None
    declared = declared.get_object() if declared is not None else None
    if node.get("/Type") != "/Pages" or not isinstance(kids, ArrayObject) or not isinstance(declared, NumberObject):
        raise ValueError("Invalid page-tree structure")  # noqa: TRY003
    actual = sum(_page_tree_count(child, seen) for child in kids)
    if actual != declared:
        raise ValueError("Page count does not match the page tree")  # noqa: TRY003
    return actual


def _stream_filters(stream: StreamObject) -> tuple[tuple[NameObject, PdfObject | None], ...]:
    filters = stream.get("/Filter")
    filters = filters.get_object() if filters is not None else None
    if filters is None or isinstance(filters, NullObject):
        filters = []
    elif not isinstance(filters, ArrayObject):
        filters = [filters]
    params = stream.get("/DecodeParms")
    params = params.get_object() if params is not None else None
    if params is None or isinstance(params, NullObject):
        params = [None] * len(filters)
    elif not isinstance(params, ArrayObject):
        params = [params]
    if len(filters) != len(params):
        raise ValueError("Stream filters and parameters do not match")  # noqa: TRY003
    result = []
    for filter_value, parameter in zip(filters, params, strict=True):
        filter_name = filter_value.get_object()
        if not isinstance(filter_name, NameObject):
            raise TypeError("Invalid stream filter")  # noqa: TRY003
        result.append((filter_name, parameter))
    return tuple(result)


def _validate_text_stream(stream: StreamObject, checked: set[int]) -> None:
    if id(stream) in checked:
        return
    checked.add(id(stream))
    # The base method reads encoded bytes. pypdf's recovery decoder may
    # silently turn damaged Flate data into an empty page, even in strict mode.
    data = StreamObject.get_data(stream)
    for filter_name, parameter in _stream_filters(stream):
        if filter_name in ("/FlateDecode", "/Fl"):
            decoder = zlib.decompressobj()
            limit = get_configuration().zlib_maximum_output_length
            length = len(decoder.decompress(data, limit + 1 if limit else 0))
            if limit and (length > limit or decoder.unconsumed_tail):
                raise ValueError("Compressed text exceeds the configured output limit")  # noqa: TRY003
            if not decoder.eof:
                raise ValueError("Truncated compressed text stream")  # noqa: TRY003
        stage = StreamObject()
        stage.set_data(data)
        stage[NameObject("/Filter")] = filter_name
        if parameter is not None:
            stage[NameObject("/DecodeParms")] = parameter
        data = decode_stream_data(stage)


def _validate_resources(resources: PdfObject | None, checked: set[int]) -> None:
    if resources is None or isinstance(resources, NullObject):
        return
    resources = resources.get_object()
    if not isinstance(resources, DictionaryObject):
        raise TypeError("Missing page resources")  # noqa: TRY003
    objects = resources.get("/XObject")
    if objects is None:
        return
    objects = objects.get_object()
    if not isinstance(objects, DictionaryObject):
        raise TypeError("Missing page XObjects")  # noqa: TRY003
    for reference in objects.values():
        form = reference.get_object()
        if not isinstance(form, StreamObject):
            raise TypeError("Missing XObject stream")  # noqa: TRY003
        if form.get("/Subtype") == "/Form" and id(form) not in checked:
            _validate_text_stream(form, checked)
            _validate_page_content(form, checked)


def _validate_page_content(page: DictionaryObject, checked: set[int]) -> None:
    contents = page.get("/Contents")
    if contents is not None:
        resolved = contents.get_object()
        if resolved is None:
            raise ValueError("Missing page content reference")  # noqa: TRY003
        streams = resolved if isinstance(resolved, ArrayObject) else [resolved]
        if not isinstance(resolved, NullObject):
            for value in streams:
                stream = value.get_object()
                if not isinstance(stream, StreamObject):
                    raise TypeError("Page content is not a stream")  # noqa: TRY003
                _validate_text_stream(stream, checked)
    _validate_resources(page.get("/Resources"), checked)


def _called_xobject(resources: PdfObject | None, operands: Sequence[PdfObject]) -> StreamObject:
    resources = resources.get_object() if resources is not None else None
    objects = resources.get("/XObject") if isinstance(resources, DictionaryObject) else None
    objects = objects.get_object() if objects is not None else None
    if len(operands) != 1 or not isinstance(operands[0], NameObject) or not isinstance(objects, DictionaryObject):
        raise ValueError("Invalid XObject invocation")  # noqa: TRY003
    reference = objects.get(operands[0])
    target = reference.get_object() if reference is not None else None
    if not isinstance(target, StreamObject):
        raise TypeError("Missing invoked XObject")  # noqa: TRY003
    return target


def _form_invocations(
    content: PdfObject | None,
    resources: PdfObject | None,
    reader: PdfReader,
    ancestors: set[int],
    counts: dict[tuple[int, int], int],
) -> int:
    """Validate executed Form paths before pypdf can recover by dropping them."""
    if content is None:
        return 0
    identity, key = id(content), (id(content), id(resources))
    if identity in ancestors:
        raise ValueError("Cyclic Form XObject invocation")  # noqa: TRY003
    if key in counts:
        return counts[key]
    ancestors.add(identity)
    total = 0
    try:
        for operands, operator in ContentStream(content, reader).operations:
            if operator != b"Do":
                continue
            target = _called_xobject(resources, operands)
            if target.get("/Subtype") == "/Form":
                total += 1 + _form_invocations(target, target.get("/Resources", resources), reader, ancestors, counts)
                if total > get_configuration().xform_maximum_invocations_per_extraction:
                    raise ValueError("Form invocation limit would truncate extracted text")  # noqa: TRY003
        counts[key] = total
        return total
    finally:
        ancestors.remove(identity)


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every page, or fail without publishing a partial document.

    Plain extraction preserves text and table values within their original page.
    Layout normalization and query expansion never modify this captured text.
    """

    digest = hashlib.sha256(value.content).hexdigest()
    try:
        with io.BytesIO(value.content) as stream:
            reader = PdfReader(stream, strict=True)
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("PDF requires a password")  # noqa: TRY003, TRY301
            catalog = reader.trailer["/Root"]
            if not isinstance(catalog, DictionaryObject):
                raise TypeError("Invalid PDF catalog")  # noqa: TRY003, TRY301
            expected = _page_tree_count(catalog["/Pages"], set())
            pages = tuple(reader.pages)
            if len(pages) != expected:
                raise ValueError("Not all physical pages could be read")  # noqa: TRY003, TRY301
            checked: set[int] = set()
            form_counts: dict[tuple[int, int], int] = {}
            result = []
            for number, page in enumerate(pages, 1):
                _validate_page_content(page, checked)
                _form_invocations(page.get("/Contents"), page.get("/Resources"), reader, set(), form_counts)
                result.append(
                    PDFPageInput(
                        document_id=value.document_id,
                        document_sha256=digest,
                        page_number=number,
                        text=page.extract_text() or "",
                    )
                )
            return tuple(result)
    except Exception as error:
        # pypdf can raise decoding, decompression and page-tree exceptions in
        # addition to PdfReadError. All are an atomic input failure at this API.
        raise ValueError("Unable to completely parse PDF") from error  # noqa: TRY003


class _PDFPageSource(Source):
    page: PDFPageInput


class _PDFPageAdapter:
    input_class = PDFPageInput
    source_class = _PDFPageSource
    name = "pdf-page"

    async def resolve(self, value: PDFPageInput, /) -> _PDFPageSource:
        # Encode the tuple, rather than concatenating ambiguous delimiters or
        # truncating caller identifiers. Text is deliberately not identity.
        identity = json.dumps(
            (value.document_id, value.document_sha256, value.page_number),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return _PDFPageSource(
            name=f"pdf-page:{hashlib.sha256(identity).hexdigest()}",
            materialization=SourceMaterialization.CAPTURED,
            page=value,
        )

    async def read(self, source: _PDFPageSource, /) -> PDFPageInput:
        return source.page


class _PDFPageProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = _PDFPageSource
    output_class = TextEvidence

    def project(self, source: _PDFPageSource, /) -> TextEvidence:
        page = source.page
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{page.document_id}\n{page.text}",
            metadata={
                "document_id": page.document_id,
                "document_sha256": page.document_sha256,
                "page_number": page.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    _PDFPageAdapter(), version="1", projections=(_PDFPageProjection(),)
)

_WORDS = re.compile(r"[a-z]+|\d+")
_FISCAL_YEAR = re.compile(r"\bfy\s*(\d{4})\b")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_STOP_WORDS = frozenset([
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "most",
    "not",
    "of",
    "on",
    "or",
    "our",
    "s",
    "shall",
    "should",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "any",
    "also",
    "answer",
    "assume",
    "assuming",
    "calculate",
    "define",
    "details",
    "directly",
    "explicitly",
    "give",
    "information",
    "only",
    "outlined",
    "perspective",
    "please",
    "question",
    "quantity",
    "reported",
    "relying",
    "respond",
    "response",
    "shown",
    "state",
    "unadjusted",
    "use",
    "using",
    "within",
    "fy",
    "usd",
    "millions",
])

# General financial vocabulary, independent of any document or benchmark item.
# Each tuple lists trigger words and short lexical probes for alternative wording.
_CONCEPTS = (
    ("revenue revenues sales turnover", ("revenue sales",)),
    ("profit profits earnings income", ("income earnings profit",)),
    (
        "geography geographies geographic geographical countries regions territories",
        (
            "geographic geographical",
            "countries regions",
            "segments international",
        ),
    ),
    ("legal litigation lawsuit lawsuits battles proceedings", ("legal proceedings", "litigation contingencies")),
    ("restructuring reorganizing reorganization", ("restructuring charges",)),
    ("ebitda", ("operating income", "operating profit", "depreciation amortization", "cash flows")),
    ("capex expenditure expenditures", ("capital spending", "capital expenditures")),
    ("margin margins", ("net revenue",)),
    ("cash cashflow flows", ("cash flows",)),
    ("debt borrowings borrowing leverage", ("debt borrowings",)),
    ("dividend dividends", ("dividends payments",)),
    ("repurchase repurchases buyback buybacks", ("share repurchases",)),
    ("employees employee workforce headcount", ("employees workforce",)),
    ("assets asset liabilities liability balance", ("balance sheets",)),
    ("stock equity shareholders stockholders", ("stockholders equity",)),
    ("research development", ("research development",)),
)

# Title matches distinguish primary statements from notes which happen to
# repeat all of their terms. Requests can use either the formal name or P&L.
_STATEMENTS = (
    (
        re.compile(
            r"\b(?:income|earnings|operations)\s+statements?\b|\bstatements?\s+of\s+(?:income|operations|earnings)\b|\bp\s*(?:&|and)\s*l\b|\bebitda\b",
            re.I,
        ),
        re.compile(r"\bstatements?\s+of\s+(?:income|operations|earnings)\b|\bprofit\s+and\s+loss\b"),
        ("statement income", "statements operations", "statements earnings"),
    ),
    (
        re.compile(r"\bcash\s*flows?\b|\bebitda\b", re.I),
        re.compile(r"\bstatements?\s+of\s+cash\s+flows?\b"),
        ("statement cash flows",),
    ),
    (
        re.compile(r"\bbalance\s+sheets?\b|\b(?:current|quick)\s+ratio\b|\bworking\s+capital\b", re.I),
        re.compile(r"\bbalance\s+sheets?\b|\bstatements?\s+of\s+financial\s+position\b"),
        ("balance sheet", "financial position"),
    ),
)


def _tokens(value: str) -> list[str]:
    value = _FISCAL_YEAR.sub(r"\1", value.lower())
    return _WORDS.findall(value)


def _stem(word: str) -> str:
    # The corpus stays untouched; lightweight inflection handling is used only
    # for scoring and never changes a citation's retrievable text.
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def _query_plan(question: str) -> tuple[str, dict[str, float]]:
    tokens = _tokens(question)
    words = tuple(dict.fromkeys(word for word in tokens if word not in _STOP_WORDS))
    weights = {_stem(word): 1.0 for word in words}
    terms = set(tokens)
    for triggers, alternatives in _CONCEPTS:
        if terms.intersection(triggers.split()):
            for word in _tokens(" ".join(alternatives)):
                weights.setdefault(_stem(word), 0.55)
    # All active pages are admitted from public Memory records. Inflection and
    # concept variants are therefore scored locally, without repeating full
    # Memory searches for every spelling. FTS supplies a single lexical signal.
    return " ".join(words), weights


_COMPANY_SUFFIXES = frozenset({"inc", "incorporated", "corp", "corporation", "company", "co", "plc", "ltd", "limited"})


def _company_words(document_id: str) -> tuple[str, ...]:
    year = _YEAR.search(document_id)
    label = document_id[: year.start()] if year is not None else document_id
    return tuple(word for word in _tokens(label) if word not in _COMPANY_SUFFIXES)


def _mentions_document(words: tuple[str, ...], question_words: tuple[str, ...]) -> bool:
    label = "".join(words)
    if len(label) < 2 or label in _STOP_WORDS:
        return False
    # Contiguous token windows accept punctuation and joined company names,
    # including short identifiers and names such as 3M, without substring
    # matches inside an unrelated company's name.
    for start in range(len(question_words)):
        for end in range(start + 1, min(len(question_words), start + len(words) + 3) + 1):
            if "".join(question_words[start:end]) == label:
                return True
    return False


def _selected_reports(matches: Mapping[str, bool], years: set[str]) -> set[str]:
    named = {identity for identity, match in matches.items() if match}
    # An explicit year can distinguish annual reports of the named company.
    # If no corresponding report exists, retain its available reports.
    annual = {identity for identity in named if years.intersection(_YEAR.findall(identity))}
    return annual or named or set(matches)


def _rank_candidates(
    hits: Sequence[MemoryEntryRecord], weights: Mapping[str, float], question: str, probes_ranks: Sequence[float]
) -> list[int]:
    if not hits:
        return []
    # Invert the scorer's inflection rule once. BM25 only needs counts for
    # query vocabulary; other tokens still contribute to the full page length.
    # Stored evidence and the normalization rule remain unchanged.
    lexemes: dict[str, str] = {}
    for stem in weights:
        forms = (stem, f"{stem}s", f"{stem[:-1]}ies" if stem.endswith("y") else stem)
        lexemes.update((form, stem) for form in forms if _stem(form) == stem)
    documents: list[tuple[str, Counter[str], int, str]] = []
    frequencies: Counter[str] = Counter()
    total_length = 0
    for hit in hits:
        document_id, _separator, body = hit.entry.text.partition("\n")
        words = _tokens(body)
        count = Counter(stem for word in words if (stem := lexemes.get(word)) is not None)
        frequencies.update(count.keys())
        total_length += len(words)
        documents.append((document_id.lower(), count, len(words), " ".join(words[:70])))
    average = total_length / len(documents) or 1.0
    weighted_idfs = {
        word: weight * math.log1p((len(documents) - frequencies[word] + 0.5) / (frequencies[word] + 0.5))
        for word, weight in weights.items()
    }
    question_words = tuple(word for word in _tokens(question) if word not in _COMPANY_SUFFIXES)
    company_words = {identity: _company_words(identity) for identity in {item[0] for item in documents}}
    company_stems = {identity: {_stem(word) for word in words} for identity, words in company_words.items()}
    document_matches = {
        identity: _mentions_document(words, question_words) for identity, words in company_words.items()
    }
    question_years = set(_YEAR.findall(question))
    matching_years = {
        identity: bool(question_years.intersection(_YEAR.findall(identity))) for identity in company_words
    }
    requested_titles = tuple(title for request, title, _probes in _STATEMENTS if request.search(question))
    scores = []
    for index, (document_id, count, length, heading) in enumerate(documents):
        score = 0.0
        normalization = 1.2 * (0.35 + 0.65 * length / average)
        for word, weighted_idf in weighted_idfs.items():
            # Company and year select a document, rather than a topic.
            if word in question_years or word in company_stems[document_id]:
                continue
            frequency = count[word]
            if frequency:
                score += weighted_idf * frequency * 2.2 / (frequency + normalization)
        title_matches = sum(
            bool(match is not None and match.start() < 100)
            for title in requested_titles
            for match in (title.search(heading),)
        )
        score = score * (1.0 + 0.5 * title_matches) + 8.0 * title_matches
        if document_matches[document_id]:
            score *= 1.8
        if matching_years[document_id]:
            score *= 1.15
        # Preserve the exact fusion weight and stable tie order.
        score += 4.0 * probes_ranks[index]
        scores.append(score)
    selected = _selected_reports(document_matches, question_years)
    eligible = [index for index, (identity, *_rest) in enumerate(documents) if identity in selected]
    return sorted(eligible, key=lambda index: (-scores[index], index))


async def search_pdf_pages(
    memory: ScopedMemoryApplication, question: str, limit: int = 10
) -> tuple[MemoryCitation, ...]:
    """Retrieve distinct physical pages through public scoped Memory APIs.

    Search explicitly selects the lexical channel; no embedding is requested.
    Returned references retain the exact Memory revision and entry version
    supplied by public Memory records, so evidence remains readable after restart.
    """

    if limit <= 0 or not question.strip():
        return ()
    query, weights = _query_plan(question)
    if not query:
        return ()
    from powercontext.builtin.runtime import SearchMemoryRequest

    pool_limit = max(80, min(limit * 8, 320))
    # The lexical channel has a finite coarse pool. Capture active records
    # through the same public scoped Memory before ranking. These references
    # identify genuine persisted versions; no external index or files are used.
    records = (await memory.list()).entries
    hits = [
        record
        for record in records
        if len(record.entry.sources) == 1 and record.entry.sources[0].source_type == "pdf-page"
    ]
    if not hits:
        return ()
    positions = {
        (hit.memory_ref.artifact_id, hit.memory_ref.revision, hit.entry.entry_id, hit.entry.entry_version_id): index
        for index, hit in enumerate(hits)
    }
    ranks = [0.0] * len(hits)
    page = await memory.search(SearchMemoryRequest(query=query, limit=pool_limit, mode="fts"))
    for rank, hit in enumerate(page.hits, 1):
        key = (hit.memory_ref.artifact_id, hit.memory_ref.revision, hit.entry_id, hit.entry_version_id)
        # Search may include non-PDF evidence, or a newer Memory revision
        # committed after list(). Score only the captured active PDF set;
        # its exact historical citations remain readable in either case.
        if key in positions:
            ranks[positions[key]] += 1.0 / (60 + rank)
    citations, seen = [], set()
    for index in _rank_candidates(hits, weights, question, ranks):
        record = hits[index]
        source_id = record.entry.sources[0].source_id
        if source_id in seen:
            continue
        seen.add(source_id)
        # Return the exact citation supplied by this public Memory record.
        # list() already read its persisted version and single-page evidence;
        # no synthetic revision or repeated full-Memory read is needed here.
        citations.append(record.citation)
        if len(citations) == limit:
            break
    return tuple(citations)


__all__ = ["PDF_PAGE_SOURCE_DEFINITION", "PDFDocumentInput", "PDFPageInput", "parse_pdf_document", "search_pdf_pages"]
