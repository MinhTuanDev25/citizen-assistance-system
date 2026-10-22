-- Allow verified (human-proofed) page text distinct from raw OCR.

ALTER TABLE knowledge_chunks
    DROP CONSTRAINT knowledge_chunks_source_check;

ALTER TABLE knowledge_chunks
    ADD CONSTRAINT knowledge_chunks_source_check
        CHECK (extraction_source IN ('native', 'ocr', 'mixed', 'verified'));
