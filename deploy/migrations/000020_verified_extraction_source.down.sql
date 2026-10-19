-- Revert verified source only when no such rows exist.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM knowledge_chunks WHERE extraction_source = 'verified') THEN
        RAISE EXCEPTION 'knowledge_chunks still has verified rows; 000020 down will not coerce them to ocr';
    END IF;
END $$;

ALTER TABLE knowledge_chunks
    DROP CONSTRAINT knowledge_chunks_source_check;

ALTER TABLE knowledge_chunks
    ADD CONSTRAINT knowledge_chunks_source_check
        CHECK (extraction_source IN ('native', 'ocr', 'mixed'));
