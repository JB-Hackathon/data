from __future__ import annotations


from jb_hackathon_data.text_cleaning import clean_text


def test_clean_text_normalizes_newlines_tabs_spaces_and_blank_lines() -> None:
    text = "제 1 조\r\n내용\t입니다   입니다.\r다음줄입니다.\n\n\n\n\n끝"
    assert clean_text(text) == "제 1 조\n내용 입니다 입니다.\n다음줄입니다.\n\n\n끝"


def test_clean_text_removes_control_chars_but_preserves_legal_symbols() -> None:
    text = "A\x00B\x1fC\t□◦➡ ①②③ ž"
    assert clean_text(text) == "ABC □◦➡ ①②③ ž"


def test_clean_text_removes_footer_like_page_markers() -> None:
    text = "본문 내용\n1\n\n- 2 -\n - 10 - \n본문2"
    assert clean_text(text) == "본문 내용\n\n본문2"


def test_clean_text_keeps_table_pipe_and_paragraph_markers() -> None:
    text = "제 1 조\n| A | B |\n|---|---|\n| 1 | 2 |\n\n③②① 2"
    assert clean_text(text) == "제 1 조\n| A | B |\n|---|---|\n| 1 | 2 |\n\n③②① 2"

