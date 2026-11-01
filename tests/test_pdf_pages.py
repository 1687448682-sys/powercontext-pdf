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

"""PDF capture and citation regressions exercised through public interfaces."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from powercontext.sources import SourceDefinitionRegistry, SourceMaterialization, project_source_for_transport
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    parse_pdf_document,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _pdf(*texts: str, password: str | None = None, broken_last_page: bool = False) -> bytes:
    writer = PdfWriter()
    for index, text in enumerate(texts):
        page = writer.add_blank_page(width=300, height=400)
        if not text:
            continue
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(
            b"BT (unterminated"
            if broken_last_page and index == len(texts) - 1
            else f"BT /F1 12 Tf 20 300 Td ({escaped}) Tj ET".encode("ascii")
        )
        page[NameObject("/Contents")] = stream
    if password is not None:
        writer.encrypt(user_password=password, owner_password="owner-secret")  # noqa: S106 -- synthetic fixture
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


def test_complete_pages_keep_numbers_blank_boundaries_and_digest():
    raw = _pdf("Revenue 123.45 expenses 67", "", "Debt 890.12")
    pages = parse_pdf_document(PDFDocumentInput(document_id="report", content=raw))
    assert isinstance(pages, tuple)
    assert [(p.page_number, p.text) for p in pages] == [(1, "Revenue 123.45 expenses 67"), (2, ""), (3, "Debt 890.12")]
    assert all(p.document_sha256 == hashlib.sha256(raw).hexdigest() for p in pages)


@pytest.mark.parametrize("document_id", ["", " \n\t"])
def test_reject_blank_document_identifier(document_id):
    with pytest.raises(ValueError):
        PDFDocumentInput(document_id=document_id, content=_pdf("value"))


@pytest.mark.parametrize("raw", [b"", b"not a PDF", b"%PDF-1.7\ntruncated", _pdf("ok", "bad", broken_last_page=True)])
def test_corruption_is_an_atomic_failure(raw):
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="report", content=raw))


def test_permissions_and_password_protection():
    readable = parse_pdf_document(PDFDocumentInput(document_id="permissions", content=_pdf("123", password="")))
    assert readable[0].text == "123"
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="locked", content=_pdf("123", password="required")))  # noqa: S106


async def test_definition_identity_projection_and_revision():
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    raw = _pdf("First page", "Second page")
    document_id = " document:with/delimiters " + "璐㈡姤" * 200
    pages = parse_pdf_document(PDFDocumentInput(document_id=document_id, content=raw))
    source = await registry.resolve(pages[0])
    assert source.materialization == SourceMaterialization.CAPTURED
    assert source.definition_version == "1" and len(source.name) <= 256
    assert (await registry.resolve(pages[0])).name == source.name
    assert (await registry.resolve(pages[1])).name != source.name
    other = pages[0].model_copy(update={"document_id": "different"})
    assert (await registry.resolve(other)).name != source.name
    assert await registry.read(source) == pages[0]
    projection = project_source_for_transport(registry, source).projection(TEXT_EVIDENCE_PROJECTION_KEY)
    assert projection == {
        "source_type": "pdf-page",
        "source_id": source.name,
        "content": f"{document_id}\nFirst page",
        "metadata": {
            "document_id": document_id,
            "document_sha256": hashlib.sha256(raw).hexdigest(),
            "page_number": 1,
        },
    }
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(raw)))
    writer.add_metadata({"/Title": "A metadata-only revision"})
    revised = io.BytesIO()
    writer.write(revised)
    updated = parse_pdf_document(PDFDocumentInput(document_id=document_id, content=revised.getvalue()))
    assert updated[0].text == pages[0].text
    assert (await registry.resolve(updated[0])).name != source.name


async def test_scoped_retrieval_survives_restart_and_preserves_exact_citations(tmp_path: Path):
    from powercontext.builtin.artifacts.memory import MemoryEntryInput
    from powercontext.builtin.persistence.sqlite import SQLiteConfig
    from powercontext.builtin.runtime import (
        BuiltinConfig,
        GetMemoryEntryRequest,
        SubmitSourceObservation,
        open_builtin_runtime,
    )
    from powercontext.builtin.scope import ScopeDraft
    from powercontext.sources import manifest_for_definition

    class Pipeline:
        async def extract(self, request, /):
            return tuple(
                MemoryEntryInput(
                    kind="source-evidence", text=s.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"], sources=(s,)
                )
                for s in request.sources
            )

    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    definition = manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION)
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'pages.db'}"))
    observations = []
    for document_id, raw in (
        (
            "ALPHA_2022_10K",
            _pdf(
                "Consolidated Statement of Income. Operating income 120 net revenue 200",
                "Consolidated Statement of Cash Flows. Depreciation amortization 30",
                "",
                "Note: EBITDA operating income earnings depreciation amortization cash flows " * 10,
                "Consolidated Balance Sheets. Current assets 150 current liabilities 75",
            ),
        ),
        (
            "ALPHA_2022_10K",
            _pdf(
                "Consolidated Statement of Income. Operating income 121 net revenue 201",
                "Consolidated Statement of Cash Flows. Depreciation amortization 31",
                "",
                "Note: EBITDA operating income earnings depreciation amortization cash flows " * 10,
                "Consolidated Balance Sheets. Current assets 160 current liabilities 80",
            ),
        ),
    ):
        for page in parse_pdf_document(PDFDocumentInput(document_id=document_id, content=raw)):
            observations.append(project_source_for_transport(registry, await registry.resolve(page)))
    async with open_builtin_runtime(config, candidate_pipeline=Pipeline()) as runtime:
        scope = await runtime.scopes.create(ScopeDraft(title="PDFs", summary="Reports", idempotency_key="pdfs"))
        other = await runtime.scopes.create(ScopeDraft(title="Other", summary="Private", idempotency_key="other"))
        await runtime.ingestion.register(definition)
        for observation in (*observations, observations[0]):
            await runtime.ingestion.submit(SubmitSourceObservation(scope_id=scope.scope_id, observation=observation))
        memory = runtime.memory.for_scope(scope.scope_id)
        while (await memory.flush()).processed:
            pass
        private_page = parse_pdf_document(
            PDFDocumentInput(document_id="PRIVATE_2022_10K", content=_pdf("Operating income depreciation amortization"))
        )[0]
        private_source = project_source_for_transport(registry, await registry.resolve(private_page))
        await runtime.ingestion.submit(SubmitSourceObservation(scope_id=other.scope_id, observation=private_source))
        private_memory = runtime.memory.for_scope(other.scope_id)
        while (await private_memory.flush()).processed:
            pass
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        assert len((await memory.list()).entries) == len(observations)
        citations = await search_pdf_pages(memory, "Alpha FY2022 EBITDA operating income depreciation", limit=4)
        assert len(citations) == 4
        found = []
        for citation in citations:
            entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
            assert len(entry.sources) == 1
            found.append(entry.sources[0].source_id)
            expected = next(s for s in observations if s.name == entry.sources[0].source_id)
            assert entry.text == expected.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"].strip()
            assert "Consolidated Statement" in entry.text
        assert len(set(found)) == 4
        assert len(await search_pdf_pages(memory, "operating income", limit=1)) == 1
        balance_citations = await search_pdf_pages(
            memory, "Alpha FY2022 balance sheet current asset current liability", limit=2
        )
        assert len(balance_citations) == 2
        for citation in balance_citations:
            balance_entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
            assert "Current assets" in balance_entry.text and "current liabilities" in balance_entry.text
        assert await search_pdf_pages(memory, "operating income", limit=-1) == ()
        assert await search_pdf_pages(memory, "operating income", limit=0) == ()
        assert await search_pdf_pages(memory, "   ", limit=20) == ()
        private_citations = await search_pdf_pages(
            runtime.memory.for_scope(other.scope_id), "operating income", limit=20
        )
        assert len(private_citations) == 1
        assert not set(found).intersection({private_source.name})
        private_entry = (
            await runtime.memory.for_scope(other.scope_id).get(GetMemoryEntryRequest(citation=private_citations[0]))
        ).entry
        assert private_entry.sources[0].source_id == private_source.name
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        for citation in citations:
            assert (await memory.get(GetMemoryEntryRequest(citation=citation))).citation == citation


@pytest.mark.parametrize("kind", ["page_count", "missing_contents", "flate", "truncated_flate", "form_flate"])
def test_damaged_late_content_does_not_become_a_blank_page(kind):
    import zlib

    from pypdf.generic import EncodedStreamObject, IndirectObject, NumberObject

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(_pdf("Complete first page", "Second page"))))
    page = writer.pages[-1]
    if kind == "page_count":
        writer._pages.get_object()[NameObject("/Count")] = NumberObject(3)
    elif kind == "missing_contents":
        page[NameObject("/Contents")] = IndirectObject(99999, 0, writer)
    else:
        stream = EncodedStreamObject()
        stream[NameObject("/Filter")] = NameObject("/FlateDecode")
        stream._data = zlib.compress(b"BT (Partial text) Tj ET")[:-4] if kind == "truncated_flate" else b"invalid zlib"
        if kind == "form_flate":
            stream[NameObject("/Subtype")] = NameObject("/Form")
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/XObject"): DictionaryObject({NameObject("/F0"): stream})
            })
        else:
            page[NameObject("/Contents")] = stream
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="corrupt", content=output.getvalue()))


def test_valid_empty_compressed_stream_and_filter_chain():
    import base64
    import zlib

    from pypdf.generic import ArrayObject, EncodedStreamObject, NullObject, NumberObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    stream = EncodedStreamObject()
    stream._data = base64.a85encode(zlib.compress(b""), adobe=True)[2:]
    stream[NameObject("/Filter")] = ArrayObject([NameObject("/ASCII85Decode"), NameObject("/FlateDecode")])
    stream[NameObject("/DecodeParms")] = ArrayObject([NullObject(), NullObject()])
    page[NameObject("/Contents")] = stream
    writer._pages.get_object()[NameObject("/Count")] = writer._add_object(NumberObject(1))
    output = io.BytesIO()
    writer.write(output)
    pages = parse_pdf_document(PDFDocumentInput(document_id="valid", content=output.getvalue()))
    assert [(page.page_number, page.text) for page in pages] == [(1, "")]


@pytest.mark.parametrize(
    "document_id,company",
    [
        ("AX_2022_10K", "AX"),
        ("ACME_INDUSTRIES_2022_10K", "Acme Industries"),
        ("3M_2022_10K", "3M"),
    ],
)
async def test_target_document_and_inflected_evidence_retrieved(tmp_path, document_id, company):
    from powercontext.builtin.artifacts.memory import MemoryEntryInput
    from powercontext.builtin.persistence.sqlite import SQLiteConfig
    from powercontext.builtin.runtime import (
        BuiltinConfig,
        GetMemoryEntryRequest,
        SubmitSourceObservation,
        open_builtin_runtime,
    )
    from powercontext.builtin.scope import ScopeDraft
    from powercontext.sources import manifest_for_definition

    class Pipeline:
        async def extract(self, request, /):
            return tuple(
                MemoryEntryInput(
                    kind="source-evidence", text=s.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"], sources=(s,)
                )
                for s in request.sources
            )

    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'competition.db'}"))
    observations = []
    for identity, data in (
        (document_id, _pdf("Revenue 100 operating income 50", "Product warranties 300")),
        ("BETA_2022_10K", _pdf(*(["Revenue operating income " * 20] * 24))),
    ):
        for page in parse_pdf_document(PDFDocumentInput(document_id=identity, content=data)):
            observations.append(project_source_for_transport(registry, await registry.resolve(page)))
    async with open_builtin_runtime(config, candidate_pipeline=Pipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Reports", summary="Two companies", idempotency_key="reports")
        )
        await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
        for observation in observations:
            await runtime.ingestion.submit(SubmitSourceObservation(scope_id=scope.scope_id, observation=observation))
        while (await runtime.memory.for_scope(scope.scope_id).flush()).processed:
            pass
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        for question, expected in (
            (f"What were {company}'s revenue and operating income in FY2022?", "Revenue 100"),
            (f"What is the current warranty reserve for {company} in FY2022?", "Product warranties 300"),
        ):
            citations = await search_pdf_pages(memory, question, limit=20)
            assert citations
            entries = [(await memory.get(GetMemoryEntryRequest(citation=citation))).entry for citation in citations]
            assert all(entry.text.startswith(document_id + "\n") for entry in entries)
            assert expected in entries[0].text


class _PagePipeline:
    async def extract(self, request, /):
        from powercontext.builtin.artifacts.memory import MemoryEntryInput

        return tuple(
            MemoryEntryInput(
                kind="source-evidence",
                text=source.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"],
                sources=(source,),
            )
            for source in request.sources
        )


async def test_pdf_evidence_survives_a_crowded_memory_search_pool(tmp_path):
    from powercontext.builtin.artifacts.memory import MemoryEntryInput
    from powercontext.builtin.persistence.sqlite import SQLiteConfig
    from powercontext.builtin.runtime import (
        BuiltinConfig,
        GetMemoryEntryRequest,
        RememberMemoryRequest,
        SearchMemoryRequest,
        SubmitSourceObservation,
        open_builtin_runtime,
    )
    from powercontext.builtin.scope import ScopeDraft
    from powercontext.sources import manifest_for_definition

    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'crowded.db'}"))
    source = project_source_for_transport(
        registry,
        await registry.resolve(
            parse_pdf_document(
                PDFDocumentInput(
                    document_id="REPORT_2022_10K",
                    content=_pdf("Consolidated Statements of Income. Interest expense 30. " + "Revenue 100 " * 50),
                )
            )[0]
        ),
    )
    async with open_builtin_runtime(config, candidate_pipeline=_PagePipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Mixed evidence", summary="PDF and other memories", idempotency_key="mixed")
        )
        await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
        await runtime.ingestion.submit(SubmitSourceObservation(scope_id=scope.scope_id, observation=source))
        memory = runtime.memory.for_scope(scope.scope_id)
        while (await memory.flush()).processed:
            pass
        await memory.remember(
            RememberMemoryRequest(
                entries=tuple(
                    MemoryEntryInput(kind="note", text=f"Unrelated note {i}. " + "Interest expense " * 40)
                    for i in range(200)
                )
            )
        )
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        coarse = await memory.search(SearchMemoryRequest(query="interest expense", limit=160, mode="fts"))
        assert len(coarse.hits) == 160
        assert all("Unrelated note" in hit.text for hit in coarse.hits)
        citations = await search_pdf_pages(memory, "What was total interest expense?", limit=20)
        assert len(citations) == 1
        entry = (await memory.get(GetMemoryEntryRequest(citation=citations[0]))).entry
        assert entry.sources[0].source_id == source.name
        assert "Interest expense 30" in entry.text


async def test_question_fiscal_year_selects_the_requested_report(tmp_path):
    from powercontext.builtin.artifacts.memory import MemoryEntryInput
    from powercontext.builtin.persistence.sqlite import SQLiteConfig
    from powercontext.builtin.runtime import (
        BuiltinConfig,
        GetMemoryEntryRequest,
        SubmitSourceObservation,
        open_builtin_runtime,
    )
    from powercontext.builtin.scope import ScopeDraft
    from powercontext.sources import manifest_for_definition

    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'fiscal.db'}"))
    inputs = []
    for year, texts in (
        (2021, ["Operating income expense " * 20] * 24),
        (2022, ["Consolidated Statements of Income. Operating income 35 expense 5"]),
    ):
        for page in parse_pdf_document(PDFDocumentInput(document_id=f"ACME_{year}_10K", content=_pdf(*texts))):
            source = project_source_for_transport(registry, await registry.resolve(page))
            inputs.append(
                MemoryEntryInput(
                    kind="source-evidence",
                    text=source.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"],
                    sources=(source,),
                )
            )
    async with open_builtin_runtime(config, candidate_pipeline=_PagePipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Annual reports", summary="Reports from two fiscal years", idempotency_key="annual")
        )
        await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
        for entry in inputs:
            await runtime.ingestion.submit(
                SubmitSourceObservation(scope_id=scope.scope_id, observation=entry.sources[0])
            )
        while (await runtime.memory.for_scope(scope.scope_id).flush()).processed:
            pass
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        citations = await search_pdf_pages(memory, "What was Acme's operating income expense in FY2022?", limit=20)
        assert len(citations) == 1
        entry = (await memory.get(GetMemoryEntryRequest(citation=citations[0]))).entry
        assert entry.text.startswith("ACME_2022_10K\n")


def _form_pdf(kind: str) -> bytes:
    from pypdf.generic import ArrayObject, NumberObject

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(_pdf("Complete first page", "second"))))
    page = writer.pages[-1]
    form = DecodedStreamObject()
    form.set_data(b"BT (unterminated" if kind == "broken_form" else b"BT /F1 12 Tf (Embedded revenue 123) Tj ET")
    form[NameObject("/Subtype")] = NameObject("/Form")
    form[NameObject("/BBox")] = ArrayObject([NumberObject(x) for x in (0, 0, 100, 100)])
    form[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): page["/Resources"]["/Font"]})
    objects = DictionaryObject({NameObject("/F0"): writer._add_object(form)})
    page[NameObject("/Resources")][NameObject("/XObject")] = objects
    if kind == "cyclic_form":
        form.set_data(b"q /F0 Do Q")
        form["/Resources"][NameObject("/XObject")] = objects
    stream = DecodedStreamObject()
    stream.set_data(b"q /F1 Do Q" if kind == "missing_xobject" else b"q /F0 Do Q q /F0 Do Q")
    page[NameObject("/Contents")] = writer._add_object(stream)
    if kind == "shared_form":
        extra = writer.add_blank_page(width=300, height=400)
        extra[NameObject("/Resources")] = page["/Resources"]
        extra[NameObject("/Contents")] = page["/Contents"]
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.mark.parametrize("kind", ["broken_form", "missing_xobject", "cyclic_form"])
def test_form_failures_cannot_silently_drop_a_late_page(kind):
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="forms", content=_form_pdf(kind)))


def test_shared_form_is_fully_extracted_for_each_invocation_and_page():
    pages = parse_pdf_document(PDFDocumentInput(document_id="forms", content=_form_pdf("shared_form")))
    assert pages[0].text == "Complete first page"
    assert len(pages) == 3
    assert [page.text.count("Embedded revenue 123") for page in pages[1:]] == [2, 2]


def test_form_invocation_budget_failure_is_atomic():
    from pypdf import apply_configuration, get_configuration

    with (
        apply_configuration(get_configuration().with_overwrites(xform_maximum_invocations_per_extraction=1)),
        pytest.raises(ValueError),
    ):
        parse_pdf_document(PDFDocumentInput(document_id="forms", content=_form_pdf("valid")))


def test_compressed_validation_honors_the_configured_memory_limit():
    import tracemalloc
    import zlib

    from pypdf import apply_configuration, get_configuration
    from pypdf.generic import EncodedStreamObject

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(_pdf("first", "second"))))
    stream = EncodedStreamObject()
    stream._data = zlib.compress(b" " * (8 * 1024 * 1024))
    stream[NameObject("/Filter")] = NameObject("/FlateDecode")
    writer.pages[-1][NameObject("/Contents")] = writer._add_object(stream)
    raw = io.BytesIO()
    writer.write(raw)
    tracemalloc.start()
    try:
        with (
            apply_configuration(get_configuration().with_overwrites(zlib_maximum_output_length=1024)),
            pytest.raises(ValueError),
        ):
            parse_pdf_document(PDFDocumentInput(document_id="compressed", content=raw.getvalue()))
        # Validation must not allocate the full 8 MiB payload before pypdf's
        # configured limit gets a chance to reject it.
        assert tracemalloc.get_traced_memory()[1] < 4 * 1024 * 1024
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("as_array", [False, True])
def test_null_content_stays_a_legitimate_blank_page(as_array):
    from pypdf.generic import ArrayObject, NullObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    page[NameObject("/Contents")] = ArrayObject([NullObject()]) if as_array else NullObject()
    output = io.BytesIO()
    writer.write(output)
    pages = parse_pdf_document(PDFDocumentInput(document_id="blank", content=output.getvalue()))
    assert [(p.page_number, p.text) for p in pages] == [(1, "")]
