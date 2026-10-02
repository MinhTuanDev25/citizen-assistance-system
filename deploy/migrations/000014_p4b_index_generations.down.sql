-- Drops P4B generation rows. Chunks written by 000014 are removed first.
ALTER TABLE procedure_version_documents DROP CONSTRAINT IF EXISTS fk_pvd_active_generation;
ALTER TABLE procedure_version_documents DROP COLUMN IF EXISTS active_generation_id;

DELETE FROM knowledge_chunks;

ALTER TABLE knowledge_chunks DROP CONSTRAINT IF EXISTS fk_knowledge_chunks_generation_xa;
ALTER TABLE knowledge_chunks DROP CONSTRAINT IF EXISTS ux_knowledge_chunks_generation_idx;
ALTER TABLE knowledge_chunks DROP CONSTRAINT IF EXISTS knowledge_chunks_page_check;
ALTER TABLE knowledge_chunks DROP CONSTRAINT IF EXISTS knowledge_chunks_token_check;
ALTER TABLE knowledge_chunks DROP CONSTRAINT IF EXISTS knowledge_chunks_source_check;
ALTER TABLE knowledge_chunks DROP CONSTRAINT IF EXISTS knowledge_chunks_text_sha_check;

ALTER TABLE knowledge_chunks
    DROP COLUMN IF EXISTS generation_id,
    DROP COLUMN IF EXISTS page_start,
    DROP COLUMN IF EXISTS page_end,
    DROP COLUMN IF EXISTS text_sha256,
    DROP COLUMN IF EXISTS token_count,
    DROP COLUMN IF EXISTS extraction_source;

ALTER TABLE knowledge_chunks
    ADD CONSTRAINT ux_knowledge_chunks_version_doc_idx UNIQUE (procedure_version_id, document_id, chunk_index);

ALTER TABLE knowledge_chunks ALTER COLUMN embedding SET NOT NULL;

DROP TABLE IF EXISTS document_index_generations;
