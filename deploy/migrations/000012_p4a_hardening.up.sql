-- 000012_p4a_hardening.up.sql
-- 000011 recomputed processing_status only for documents that already had a
-- link, so a READY document with no links stayed READY. Recompute every row.

UPDATE documents AS doc
SET processing_status = CASE
        WHEN NOT EXISTS (
            SELECT 1 FROM procedure_version_documents AS link
            WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id
        ) THEN 'UPLOADED'
        WHEN NOT EXISTS (
            SELECT 1 FROM procedure_version_documents AS link
            WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id
              AND link.index_status <> 'READY'
        ) THEN 'READY'
        WHEN EXISTS (
            SELECT 1 FROM procedure_version_documents AS link
            WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id
              AND link.index_status = 'PROCESSING'
        ) THEN 'PROCESSING'
        WHEN EXISTS (
            SELECT 1 FROM procedure_version_documents AS link
            WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id
              AND link.index_status = 'FAILED'
        ) THEN 'FAILED'
        ELSE 'UPLOADED'
    END,
    updated_at = now();
