# data

ComplianceJB 프로젝트의 PostgreSQL/pgvector 기반 데이터베이스와 RAG 참고 문서 데이터를 관리하는 폴더입니다.

## 1. 서비스 DB ERD

서비스 DB는 DrawDB에서 설계한 ERD를 기준으로 구성합니다.

- ERD 이미지: `erd/ComplianceJB_2026-06-05T02_19_21.485Z.png`
- DB 스키마: `init/01-schema.sql`
- pgvector 확장: `init/00-extension.sql`
- 검색 인덱스: `init/02-indexes.sql`

주요 테이블은 다음과 같습니다.

| 테이블 | 역할 |
| --- | --- |
| `users` | 사용자 계정 정보 |
| `teams` | 사용자 소속 팀 |
| `review_boards` | 심의 게시물 단위 |
| `review_content_versions` | 버전별 심의 콘텐츠 |
| `review_content_chunks` | 심의 콘텐츠 RAG 검색용 청크/임베딩 |
| `reference_documents` | 준법 자문 참고 문서 |
| `reference_document_chunks` | 참고 문서 RAG 검색용 청크/임베딩 |

## 2. DB 구축

### 2.1 참고 문서 청킹 및 임베딩

원본 참고 문서는 `raw/`에 저장합니다.

```text
raw/
  *.pdf
  *.hwp
  reference_documents.json
```

문서별 메타데이터는 `raw/reference_documents.json`에서 관리합니다.

- `document_type`
- `title`
- `issuing_authority`
- `issued_date`
- `source_file_path`
- `metadata.sections`

추출 및 청킹 산출물은 다음 위치에 저장합니다.

```text
outputs/extraction/raw_documents.jsonl
outputs/chunking/rag_chunks.jsonl
outputs/chunking/summary.json
```

원본 문서를 다시 추출하려면 다음 명령을 사용합니다.

```bash
uv run python -m jb_hackathon_data.run_clean_pipeline \
  --raw-dir raw \
  --out outputs/extraction/raw_documents.jsonl
```

추출 결과를 다시 청킹하려면 다음 명령을 사용합니다.

```bash
uv run python -m jb_hackathon_data.run_chunk_pipeline \
  --results outputs/extraction/raw_documents.jsonl \
  --out outputs/chunking/rag_chunks.jsonl \
  --summary-out outputs/chunking/summary.json
```

현재 DB 적재 스크립트는 `outputs/chunking/rag_chunks.jsonl`에 포함된 `embedding` 값을 사용합니다.

### 2.2 PostgreSQL DB 초기 구축 및 Reference Document 데이터 추가

초기 프로젝트 구축 시에는 PostgreSQL 컨테이너를 먼저 생성한 뒤 참고 문서 데이터를 적재합니다.

```bash
docker compose up --build -d
```

컨테이너가 처음 생성될 때 `init/`의 SQL 파일이 실행됩니다.

```text
init/
  00-extension.sql  # pgvector extension 생성
  01-schema.sql     # enum/table 생성
  02-indexes.sql    # vector/full-text search index 생성
```

DB가 정상적으로 올라온 뒤 참고 문서와 문서 청크 데이터를 적재합니다.

```bash
uv run python -m jb_hackathon_data.db.load_reference_data --reset
```

`--reset` 옵션은 기존 `reference_documents`, `reference_document_chunks` 데이터를 초기화한 뒤 다시 적재합니다.

초기화 없이 upsert만 하려면 다음처럼 실행합니다.

```bash
uv run python -m jb_hackathon_data.db.load_reference_data
```

기본 접속 정보는 다음과 같습니다.

```text
DATABASE_URL=postgresql://jbuser:jbpass@localhost:5432/jbdb
```

### 2.3 Reference Document Hybrid Search 샘플

사용자 입력 쿼리를 기준으로 `reference_document_chunks`에서 벡터 검색과 `tsvector` 검색을 함께 수행하고, RRF로 결과를 결합하는 샘플 코드는 다음 파일에 있습니다.

```text
src/jb_hackathon_data/db/rag_search_sample.py
```

실행 예시는 다음과 같습니다.

```bash
uv run python -m jb_hackathon_data.db.rag_search_sample \
  "대출 광고에서 최저금리를 표시할 때 유의할 점" \
  --top-k 5 \
  --show-prompt
```

특정 문서 유형이나 발행기관으로 검색 범위를 좁힐 수도 있습니다.

```bash
uv run python -m jb_hackathon_data.db.rag_search_sample \
  "ETF 광고에서 투자 위험을 어떻게 표시해야 하나요?" \
  --document-type examples \
  --issuing-authority 금융감독원
```

첫 실행 시에는 쿼리 임베딩 모델을 내려받거나 로드하느라 시간이 걸릴 수 있습니다.

## 3. DB 실행

이미 DB가 생성되어 있는 상태에서는 기존 컨테이너를 실행합니다.

```bash
docker compose up -d
```

컨테이너 상태를 확인합니다.

```bash
docker compose ps
```

DB에 접속합니다.

```bash
docker exec -it jb-data-postgres psql -U jbuser -d jbdb
```

테이블 생성 여부를 확인합니다.

```sql
\dt
```

참고 문서 적재 결과를 확인합니다.

```sql
SELECT document_type, count(*)
FROM reference_documents
GROUP BY document_type
ORDER BY document_type;

SELECT count(*)
FROM reference_document_chunks;
```

컨테이너를 중지하려면 다음 명령을 사용합니다.

```bash
docker compose down
```

DB volume까지 완전히 초기화하고 다시 구축하려면 2.2 절차를 다시 수행합니다.

```bash
docker compose down -v
```
