CREATE INDEX IF NOT EXISTS idx_review_content_chunks_embedding
ON review_content_chunks
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_reference_document_chunks_embedding
ON reference_document_chunks
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_review_content_chunks_search_vector
ON review_content_chunks
USING gin (search_vector);

CREATE INDEX IF NOT EXISTS idx_reference_document_chunks_search_vector
ON reference_document_chunks
USING gin (search_vector);