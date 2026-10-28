package drafting

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var ErrSourceNotFound = errors.New("drafting: verified source not found")
var ErrUnverifiedSource = errors.New("drafting: source is not a READY generation")

// ErrSourceTooLarge means the chosen page range holds more excerpts than one
// draft request may carry. Admins pick a shorter range of the same generation.
var ErrSourceTooLarge = errors.New("drafting: source page range is too large")

// MaxDraftExcerpts and MaxDraftContentBytes mirror the AI service request limits
// (app/models/draft.py MAX_EXCERPTS and MAX_REQUEST_BODY_BYTES, minus JSON headroom).
const (
	MaxDraftExcerpts     = 80
	MaxDraftContentBytes = 200 * 1024
)

func checkDraftSize(snap SourceSnapshot) error {
	size := 0
	for _, item := range snap.Excerpts {
		size += len(item.Content)
	}
	if len(snap.Excerpts) > MaxDraftExcerpts || size > MaxDraftContentBytes {
		return fmt.Errorf("%w: pages %d-%d have %d excerpts (max %d); choose a shorter page range",
			ErrSourceTooLarge, snap.SourcePageStart, snap.SourcePageEnd, len(snap.Excerpts), MaxDraftExcerpts)
	}
	return nil
}

type SourceStore struct {
	Pool *pgxpool.Pool
}

func (s *SourceStore) Resolve(ctx context.Context, xaID string, procedureID, documentID, generationID uuid.UUID, pageStart, pageEnd int) (SourceSnapshot, error) {
	// No implicit default window: a caller must name the pages it drafts from.
	if pageStart <= 0 || pageEnd <= 0 || pageEnd < pageStart {
		return SourceSnapshot{}, fmt.Errorf("drafting: invalid page range: source_page_start and source_page_end are required")
	}
	if generationID == uuid.Nil {
		var found uuid.UUID
		err := s.Pool.QueryRow(ctx, `
			SELECT g.id
			FROM document_index_generations g
			JOIN knowledge_chunks c ON c.generation_id = g.id
			JOIN procedures p ON p.id = c.procedure_id
			WHERE g.document_id = $1
			  AND g.status = 'READY'
			  AND p.id = $2
			  AND p.xa_id = $3
			  AND c.page_start >= $4
			  AND c.page_end <= $5
			GROUP BY g.id
			ORDER BY MAX(g.published_at) DESC NULLS LAST, MAX(g.created_at) DESC
			LIMIT 1`,
			documentID, procedureID, xaID, pageStart, pageEnd,
		).Scan(&found)
		if errors.Is(err, pgx.ErrNoRows) {
			return SourceSnapshot{}, ErrSourceNotFound
		}
		if err != nil {
			return SourceSnapshot{}, err
		}
		generationID = found
	}

	snap, err := s.header(ctx, xaID, procedureID, documentID, generationID)
	if err != nil {
		return SourceSnapshot{}, err
	}
	snap.SourcePageStart = pageStart
	snap.SourcePageEnd = pageEnd
	if err := s.loadChunks(ctx, &snap, `AND page_start >= $5 AND page_end <= $6`, pageStart, pageEnd); err != nil {
		return SourceSnapshot{}, err
	}
	return snap, nil
}

// ChunkRange is an inclusive knowledge_chunks.chunk_index range of one generation.
type ChunkRange struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// ResolveChunks loads the verified excerpts of the given chunk ranges. No ranges
// means the whole generation (used to find procedure sections, never sent to a model).
func (s *SourceStore) ResolveChunks(ctx context.Context, xaID string, procedureID, documentID, generationID uuid.UUID, ranges []ChunkRange) (SourceSnapshot, error) {
	if generationID == uuid.Nil {
		return SourceSnapshot{}, ErrSourceNotFound
	}
	snap, err := s.header(ctx, xaID, procedureID, documentID, generationID)
	if err != nil {
		return SourceSnapshot{}, err
	}
	if len(ranges) == 0 {
		err = s.loadChunks(ctx, &snap, ``)
	} else {
		starts := make([]int32, 0, len(ranges))
		ends := make([]int32, 0, len(ranges))
		for _, r := range ranges {
			if r.Start < 0 || r.End < r.Start {
				return SourceSnapshot{}, fmt.Errorf("drafting: invalid chunk range")
			}
			starts = append(starts, int32(r.Start))
			ends = append(ends, int32(r.End))
		}
		err = s.loadChunks(ctx, &snap, `AND EXISTS (
			SELECT 1 FROM unnest($5::int[], $6::int[]) AS r(lo, hi)
			WHERE chunk_index BETWEEN r.lo AND r.hi)`, starts, ends)
	}
	if err != nil {
		return SourceSnapshot{}, err
	}
	snap.SourcePageStart, snap.SourcePageEnd = excerptPageBounds(snap.Excerpts)
	return snap, nil
}

func (s *SourceStore) header(ctx context.Context, xaID string, procedureID, documentID, generationID uuid.UUID) (SourceSnapshot, error) {
	snap := SourceSnapshot{
		ProcedureID:  procedureID,
		DocumentID:   documentID,
		GenerationID: generationID,
		XaID:         xaID,
	}
	var generationSHA string
	err := s.Pool.QueryRow(ctx, `
		SELECT p.procedure_code, p.name, p.xa_id, d.checksum,
		       g.status, g.procedure_version_id, g.source_sha256,
		       pv.version, pv.status
		FROM document_index_generations g
		JOIN documents d ON d.id = g.document_id AND d.xa_id = g.xa_id
		JOIN procedure_versions pv ON pv.id = g.procedure_version_id
		JOIN procedures p ON p.id = pv.procedure_id
		WHERE g.id = $1 AND g.document_id = $2 AND p.id = $3 AND p.xa_id = $4`,
		generationID, documentID, procedureID, xaID,
	).Scan(
		&snap.ProcedureCode, &snap.ProcedureName, &snap.XaID, &snap.DocumentSHA256,
		&snap.GenerationStatus, &snap.GenerationVersionID, &generationSHA,
		&snap.GenerationVersion, &snap.GenerationPVStatus,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return SourceSnapshot{}, ErrSourceNotFound
	}
	if err != nil {
		return SourceSnapshot{}, err
	}
	if snap.GenerationStatus != "READY" {
		return SourceSnapshot{}, ErrUnverifiedSource
	}
	if generationSHA != "" && snap.DocumentSHA256 != "" && generationSHA != snap.DocumentSHA256 {
		return SourceSnapshot{}, fmt.Errorf("%w: document checksum does not match generation source_sha256", ErrUnverifiedSource)
	}
	return snap, nil
}

// loadChunks appends the hash-verified chunks matching filter ($5.. are extra args).
func (s *SourceStore) loadChunks(ctx context.Context, snap *SourceSnapshot, filter string, extra ...any) error {
	args := append([]any{snap.GenerationID, snap.DocumentID, snap.ProcedureID, snap.XaID}, extra...)
	rows, err := s.Pool.Query(ctx, `
		SELECT id, chunk_index, content, page_start, page_end, text_sha256, generation_id, document_id
		FROM knowledge_chunks
		WHERE generation_id = $1
		  AND document_id = $2
		  AND procedure_id = $3
		  AND xa_id = $4
		  `+filter+`
		ORDER BY chunk_index`, args...)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var item Excerpt
		if err := rows.Scan(
			&item.ExcerptID, &item.ChunkIndex, &item.Content, &item.PageStart, &item.PageEnd,
			&item.TextSHA256, &item.GenerationID, &item.DocumentID,
		); err != nil {
			return err
		}
		sum := sha256.Sum256([]byte(item.Content))
		digest := hex.EncodeToString(sum[:])
		if item.TextSHA256 != digest {
			return fmt.Errorf("%w: chunk hash mismatch", ErrUnverifiedSource)
		}
		snap.Excerpts = append(snap.Excerpts, item)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if len(snap.Excerpts) == 0 {
		return ErrSourceNotFound
	}
	return nil
}

func excerptPageBounds(items []Excerpt) (start, end int) {
	for _, ex := range items {
		if start == 0 || ex.PageStart < start {
			start = ex.PageStart
		}
		if ex.PageEnd > end {
			end = ex.PageEnd
		}
	}
	return start, end
}

type SourceOption struct {
	DocumentID     uuid.UUID `json:"document_id"`
	GenerationID   uuid.UUID `json:"generation_id"`
	Title          string    `json:"title"`
	Version        string    `json:"version"`
	VersionStatus  string    `json:"version_status"`
	PageStart      int       `json:"suggested_page_start"`
	PageEnd        int       `json:"suggested_page_end"`
	ChunkCount     int       `json:"chunk_count"`
	DocumentSHA256 string    `json:"document_sha256"`
}

func (s *SourceStore) ListReady(ctx context.Context, xaID string, procedureID uuid.UUID) ([]SourceOption, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT g.document_id, g.id, COALESCE(d.title, ''), pv.version, pv.status,
		       MIN(c.page_start), MAX(c.page_end),
		       COUNT(c.id), COALESCE(d.checksum, '')
		FROM document_index_generations g
		JOIN documents d ON d.id = g.document_id
		JOIN procedure_versions pv ON pv.id = g.procedure_version_id
		JOIN procedures p ON p.id = pv.procedure_id
		JOIN knowledge_chunks c ON c.generation_id = g.id AND c.procedure_id = p.id
		WHERE p.id = $1 AND p.xa_id = $2 AND g.status = 'READY'
		GROUP BY g.document_id, g.id, d.title, pv.version, pv.status, d.checksum
		ORDER BY MAX(g.published_at) DESC NULLS LAST`,
		procedureID, xaID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]SourceOption, 0)
	for rows.Next() {
		var item SourceOption
		if err := rows.Scan(
			&item.DocumentID, &item.GenerationID, &item.Title, &item.Version, &item.VersionStatus,
			&item.PageStart, &item.PageEnd, &item.ChunkCount, &item.DocumentSHA256,
		); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}
