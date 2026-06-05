# data

ComplianceJB 프로젝트의 PostgreSQL/pgvector 기반 데이터베이스와 RAG 참고 문서 데이터를 관리하는 폴더입니다.

이 폴더에는 별도 `compose.yml`을 두지 않고, 프로젝트 루트의 `docker-compose.yml`에 정의된 `postgres-db` 서비스를 사용합니다.

```text
JB-Hackathon/
  docker-compose.yml
  data/
    init/
    raw/
    outputs/
    src/
```

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
| `chat_threads` | 심의 콘텐츠 버전별 채팅 thread |
| `chat_messages` | 채팅 메시지 |

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
outputs/chunking/rag_chunks.raw.jsonl
outputs/chunking/rag_chunks.jsonl
outputs/chunking/summary.json
```

원본 문서를 다시 추출하려면 `data` 폴더에서 다음 명령을 실행합니다.

```bash
cd ~/JB-Hackathon/data

uv run python -m jb_hackathon_data.run_clean_pipeline \
  --raw-dir raw \
  --out outputs/extraction/raw_documents.jsonl
```

추출 결과를 다시 청킹하려면 다음 명령을 실행합니다.

```bash
uv run python -m jb_hackathon_data.run_chunk_pipeline \
  --results outputs/extraction/raw_documents.jsonl \
  --out outputs/chunking/rag_chunks.raw.jsonl \
  --summary-out outputs/chunking/summary.json
```

DB 적재 전에는 청크 파일에 임베딩이 포함되어 있어야 합니다.

이미 `outputs/chunking/rag_chunks.jsonl`에 `embedding` 필드가 포함되어 있다면 임베딩 생성 단계는 생략해도 됩니다.

위의 청킹 명령을 다시 실행해서 `rag_chunks.raw.jsonl`을 새로 만든 경우에는 다음 명령으로 임베딩을 추가해 최종 `rag_chunks.jsonl`을 생성합니다.

```bash
uv run python -m jb_hackathon_data.run_embed_pipeline \
  --input outputs/chunking/rag_chunks.raw.jsonl \
  --out outputs/chunking/rag_chunks.jsonl
```

현재 DB 적재 스크립트는 `outputs/chunking/rag_chunks.jsonl`에 포함된 `embedding` 값을 사용합니다. 임베딩 모델은 기본값으로 Google `gemini-embedding-2`를 사용하며, API 키는 gitignore 처리된 `.env`의 `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY`에서 읽습니다.

### 2.2 PostgreSQL DB 초기 구축 및 Reference Document 데이터 추가

초기 프로젝트 구축 시에는 프로젝트 루트의 `postgres-db` 서비스만 먼저 실행합니다.

```bash
cd ~/JB-Hackathon

docker compose up --build -d postgres-db
```

루트 `docker-compose.yml`의 `postgres-db` 서비스는 다음 data 폴더를 사용합니다.

```yaml
volumes:
  - ./data/postgres_data:/var/lib/postgresql/data
  - ./data/init:/docker-entrypoint-initdb.d
```

컨테이너가 처음 생성될 때 `data/init/`의 SQL 파일이 순서대로 실행됩니다.

```text
init/
  00-extension.sql  # pgvector extension 생성
  01-schema.sql     # enum/table 생성
  02-indexes.sql    # vector/full-text search index 생성
```

DB가 정상적으로 올라온 뒤 `data` 폴더에서 참고 문서와 문서 청크 데이터를 적재합니다.

```bash
cd ~/JB-Hackathon/data

uv run python -m jb_hackathon_data.db.load_reference_data --reset
```

`--reset` 옵션은 기존 `reference_documents`, `reference_document_chunks` 데이터를 초기화한 뒤 다시 적재합니다.

초기화 없이 upsert만 하려면 다음처럼 실행합니다.

```bash
uv run python -m jb_hackathon_data.db.load_reference_data
```

로컬 WSL에서 loader를 실행할 때 기본 접속 정보는 다음과 같습니다.

```text
DATABASE_URL=postgresql://jbuser:jbpass@localhost:5432/jbdb
```

Docker Compose 내부의 다른 서비스에서 DB에 접속할 때는 host를 `postgres-db`로 사용합니다.

```text
DATABASE_URL=postgresql://jbuser:jbpass@postgres-db:5432/jbdb
```

### 2.3 Reference Document Hybrid Search 샘플

사용자 입력 쿼리를 기준으로 `reference_document_chunks`에서 벡터 검색과 `tsvector` 검색을 함께 수행하고, RRF로 결과를 결합하는 샘플 코드는 다음 파일에 있습니다.

```text
src/jb_hackathon_data/db/rag_search_sample.py
```

실행 예시는 다음과 같습니다.

```bash
cd ~/JB-Hackathon/data

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

이미 DB가 생성되어 있는 상태에서는 프로젝트 루트에서 `postgres-db` 서비스만 실행합니다.

```bash
cd ~/JB-Hackathon

docker compose up -d postgres-db
```

컨테이너 상태를 확인합니다.

```bash
docker compose ps postgres-db
```

DB에 접속합니다.

```bash
docker exec -it postgres-db-container psql -U jbuser -d jbdb
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

컨테이너를 중지하려면 루트에서 다음 명령을 사용합니다.

```bash
docker compose down
```

DB 데이터를 완전히 초기화하려면 `data/postgres_data`를 삭제한 뒤 다시 실행합니다.

```bash
cd ~/JB-Hackathon

docker compose down
sudo rm -rf ./data/postgres_data
docker compose up --build -d postgres-db
```

주의: `data/postgres_data/`는 로컬 DB volume 디렉터리이므로 Git에 올리지 않습니다.
