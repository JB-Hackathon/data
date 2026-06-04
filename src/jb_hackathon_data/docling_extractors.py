"""Docling-backed structure extraction."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Final

from docling_core.types.doc.document import DocItem, DoclingDocument, TableItem
from docling_core.types.doc.labels import DocItemLabel

from .io_utils import project_root_from_package, relative_source_path, sha256_file, source_id_from_sha
from .models import BlockType, BoundingBox, DocumentBlock, ExtractedDocument

if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import PdfPipelineOptions

DOCLING_EXTENSIONS: Final = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }
)

_LABEL_TO_BLOCK_TYPE: Final[dict[DocItemLabel, BlockType]] = {
    DocItemLabel.TITLE: "title",
    DocItemLabel.SECTION_HEADER: "heading",
    DocItemLabel.TEXT: "paragraph",
    DocItemLabel.LIST_ITEM: "list",
    DocItemLabel.TABLE: "table",
    DocItemLabel.PICTURE: "image",
    DocItemLabel.CHART: "image",
    DocItemLabel.PAGE_HEADER: "footer",
    DocItemLabel.PAGE_FOOTER: "footer",
    DocItemLabel.FOOTNOTE: "footer",
}


def is_docling_supported_extension(extension: str) -> bool:
    return extension.lower() in DOCLING_EXTENSIONS


def extract_docling_document(path: Path, *, sidecar_dir: Path | None = None) -> ExtractedDocument:
    sha256 = sha256_file(path)
    source_id = source_id_from_sha(sha256)
    try:
        document = _convert_with_docling(path, do_ocr=False)
        blocks = canonical_blocks_from_docling(document, source_path=relative_source_path(path))
        text = "\n".join(_block_text(block) for block in blocks if _block_text(block)).strip()
        if not text and path.suffix.lower() == ".pdf":
            document = _convert_with_docling(path, do_ocr=True)
            blocks = canonical_blocks_from_docling(document, source_path=relative_source_path(path))
            text = "\n".join(_block_text(block) for block in blocks if _block_text(block)).strip()
        if not text:
            text = document.export_to_markdown().strip()
        docling_document_path = _save_docling_sidecar(document, source_id, sidecar_dir) if text and blocks else None
        return ExtractedDocument(
            source_id=source_id,
            source_path=relative_source_path(path),
            source_sha256=sha256,
            extension=path.suffix.lower(),
            route="docling_structured",
            parser_name="docling",
            parser_version=version("docling"),
            status="success" if text and blocks else "failed",
            text=text,
            pages_or_sections=(),
            error=None if text and blocks else "docling returned empty structured blocks",
            blocks=blocks,
            docling_document_path=docling_document_path,
        )
    except Exception as exc:  # noqa: BLE001, BROAD_EXCEPT_OK
        return ExtractedDocument(
            source_id=source_id,
            source_path=relative_source_path(path),
            source_sha256=sha256,
            extension=path.suffix.lower(),
            route="docling_structured",
            parser_name="docling",
            parser_version=_optional_version("docling"),
            status="failed",
            text="",
            pages_or_sections=(),
            error=f"{type(exc).__name__}: {exc}",
            blocks=(),
        )


def canonical_blocks_from_docling(document: DoclingDocument, *, source_path: str) -> tuple[DocumentBlock, ...]:
    blocks: list[DocumentBlock] = []
    heading_stack: list[str] = []
    for index, item_and_level in enumerate(document.iterate_items()):
        item, level = item_and_level
        if not isinstance(item, DocItem):
            continue
        block_type = _block_type(item.label)
        text = _text_for_item(item, document)
        if not text and block_type != "image":
            continue
        if block_type in {"title", "heading"}:
            _replace_heading_stack(heading_stack, level, text)
        blocks.append(
            DocumentBlock(
                block_id=f"{source_path}:{index}",
                type=block_type,
                text=text,
                page_no=_page_no(item),
                bbox=_bbox(item),
                level=level,
                section_path=tuple(heading_stack),
                table_markdown=text if block_type == "table" else None,
                metadata=_metadata(item),
            )
        )
    return tuple(blocks)


def _convert_with_docling(path: Path, *, do_ocr: bool) -> DoclingDocument:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.XLSX,
            InputFormat.HTML,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options(do_ocr=do_ocr),
            ),
        },
    )
    return converter.convert(path).document


def pdf_pipeline_options(*, do_ocr: bool) -> PdfPipelineOptions:
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    return PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=True)


def _replace_heading_stack(stack: list[str], level: int, text: str) -> None:
    if level <= 0:
        stack.clear()
        stack.append(text)
        return
    del stack[level - 1 :]
    stack.append(text)


def _block_type(label: DocItemLabel) -> BlockType:
    return _LABEL_TO_BLOCK_TYPE.get(label, "unknown")


def _text_for_item(item: DocItem, document: DoclingDocument) -> str:
    if isinstance(item, TableItem):
        return item.export_to_markdown(doc=document).strip()
    text = getattr(item, "text", "")
    if isinstance(text, str):
        return text.strip()
    return ""


def _page_no(item: DocItem) -> int | None:
    if not item.prov:
        return None
    return item.prov[0].page_no


def _bbox(item: DocItem) -> BoundingBox | None:
    if not item.prov:
        return None
    bbox = item.prov[0].bbox
    return BoundingBox(
        left=float(bbox.l),
        top=float(bbox.t),
        right=float(bbox.r),
        bottom=float(bbox.b),
    )


def _metadata(item: DocItem) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = [("docling_label", item.label.value)]
    if item.self_ref:
        values.append(("docling_ref", item.self_ref))
    if item.prov:
        values.append(("bbox_origin", item.prov[0].bbox.coord_origin.value))
    return tuple(values)


def _block_text(block: DocumentBlock) -> str:
    return block.table_markdown or block.text


def _save_docling_sidecar(document: DoclingDocument, source_id: str, sidecar_dir: Path | None) -> str | None:
    if sidecar_dir is None:
        return None
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_dir / f"{source_id}.json"
    document.save_as_json(sidecar_path)
    return _path_reference(sidecar_path)


def _path_reference(path: Path) -> str:
    root = project_root_from_package().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _optional_version(package: str) -> str:
    try:
        return version(package)
    except Exception:  # noqa: BLE001, BROAD_EXCEPT_OK
        return "not-installed"
