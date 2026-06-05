# 청킹 실험 요약

JB금융그룹 준법심의 AI 에이전트의 RAG 적재 직전 산출물을 만들기 위한 문서 추출-청킹 파이프라인 실험 결과입니다.

## 결론

- 원본 `raw/` 31개 문서를 기준으로 전체 파이프라인을 재실행했습니다.
- 모든 문서가 추출에 성공했습니다.
- 최종 RAG 청크는 911개입니다.
- 검색/임베딩용 필드는 `embedding_text`가 아니라 `search_text`로 정리했습니다.
- `text`는 원문 청크 보존용이고, `search_text`는 검색 품질 개선용입니다.

## 현재 산출물

- 추출 결과: `outputs/extraction/raw_documents.jsonl`
- Docling sidecar: `outputs/extraction/docling_documents/`
- 최종 청크: `outputs/chunking/rag_chunks.jsonl`
- 요약 수치: `outputs/chunking/summary.json`

```json
{
  "documents": 31,
  "success_documents": 31,
  "chunks_before_filter": 941,
  "filtered_chunks": 30,
  "chunks": 911,
  "docling_chunks": 548,
  "fallback_chunks": 363,
  "max_tokens": 2048,
  "overlap_tokens": 0
}
```

## 방식

1. PDF 문서는 Docling 추출과 Docling `HybridChunker`를 우선 사용했습니다.
2. Docling sidecar가 없는 HWP 문서는 추출된 block 기반 fallback 청킹을 사용했습니다.
3. 순수 이미지 placeholder와 저정보량 청크는 가벼운 기본 필터로 제거했습니다.
4. 청크 크기, overlap, chunk_id 생성, provenance 정책은 유지했습니다.
5. 검색 품질 개선을 위해 `search_text`에만 문서명과 섹션 정보를 추가했습니다.

## `search_text` 규칙

`text` 원문은 바꾸지 않고, 검색용 `search_text`만 아래 형식으로 생성합니다.

```text
문서명: {문서명}
섹션: {heading_path}
조항/섹션명: {heading_path 마지막 값}
내용:
{text}
```

`heading_path`가 비어 있으면 섹션/조항 줄은 생략합니다.

## 검증 결과

- `search_text` 누락: 0건
- `embedding_text` 잔존: 0건
- `search_text`의 `내용:` 뒤 본문과 `text` 원문 불일치: 0건
- provenance 누락: 0건
- 2048 토큰 초과 청크: 0건
- 순수 이미지 placeholder 청크: 0건
- `uv run pytest`: 44 passed
- `uv run ruff check .`: pass
- `uv run basedpyright`: 0 errors

## 운영 메모

- Vector DB 임베딩 대상은 `search_text`를 사용합니다.
- 답변 근거 표시와 원문 확인에는 `text`, `page_refs`, `bbox_refs`, `docling_refs`, `block_ids`를 사용합니다.
- `source_path`, `token_count`, provenance 필드는 `search_text`에 넣지 않습니다.
- 확장자는 `.hwp`지만 실제 내용이 HWPML/XML인 파일은 현재 parser 대상이 아니므로 동일 문서의 PDF본을 사용했습니다.
