from jb_hackathon_data.models import ExtractedDocument


def test_import_extracted_document() -> None:
    assert ExtractedDocument.__name__ == "ExtractedDocument"

