package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	cleanupLease       = 120 * time.Second
	maxCleanupAttempts = 5
	defaultBatchSize   = 8
	maxBatchSize       = 32
	claimSQL           = `
		WITH picked AS (
			SELECT g.id
			FROM document_index_generations AS g
			WHERE g.status IN ('STAGING', 'FAILED')
			  AND g.updated_at < now() - make_interval(secs => $1)
			  AND g.cleanup_attempts < $2
			  AND (
				g.cleanup_status IN ('PENDING', 'RETRYABLE_FAILED')
				OR (
					g.cleanup_status = 'CLAIMED'
					AND g.cleanup_claim_expires_at IS NOT NULL
					AND g.cleanup_claim_expires_at <= now()
				)
			  )
			  AND NOT EXISTS (
				SELECT 1 FROM procedure_version_documents link
				WHERE link.active_generation_id = g.id
			  )
			  AND NOT EXISTS (
				SELECT 1 FROM document_index_jobs job
				WHERE job.id = g.job_id AND job.status = 'CLAIMED' AND job.claim_expires_at > now()
			  )
			ORDER BY g.updated_at
			LIMIT $5
			FOR UPDATE OF g SKIP LOCKED
		)
		UPDATE document_index_generations AS gen
		SET status = CASE WHEN gen.status = 'STAGING' THEN 'FAILED' ELSE gen.status END,
			error_code = CASE WHEN gen.status = 'STAGING' THEN 'orphan_expired' ELSE gen.error_code END,
			cleanup_status = 'CLAIMED',
			cleanup_claim_token = $3,
			cleanup_claim_expires_at = now() + make_interval(secs => $4),
			cleanup_attempts = gen.cleanup_attempts + 1,
			cleanup_error = NULL,
			updated_at = now()
		FROM picked
		WHERE gen.id = picked.id
		RETURNING gen.id::text, gen.cleanup_attempts`
	renewSQL = `
		UPDATE document_index_generations
		SET cleanup_claim_expires_at = now() + make_interval(secs => $3),
			updated_at = now()
		WHERE id = $1::uuid
		  AND cleanup_claim_token = $2
		  AND cleanup_status = 'CLAIMED'`
)

var errCleanupLost = errors.New("cleanup_lost")

func main() {
	if err := run(context.Background(), os.Getenv, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "reaper_failed code=%s\n", publicCode(err))
		os.Exit(1)
	}
}

type reaperConfig struct {
	databaseURL string
	qdrantURL   string
	collection  string
	grace       int
	batch       int
}

func loadConfig(getenv func(string) string) (reaperConfig, error) {
	cfg := reaperConfig{
		databaseURL: getenv("DATABASE_URL"),
		qdrantURL:   getenv("QDRANT_URL"),
		collection:  getenv("QDRANT_COLLECTION"),
		grace:       120,
		batch:       defaultBatchSize,
	}
	if cfg.databaseURL == "" || cfg.qdrantURL == "" || cfg.collection == "" {
		return reaperConfig{}, errors.New("config_missing")
	}
	if raw := getenv("INDEX_ORPHAN_GRACE_SECONDS"); raw != "" {
		if _, err := fmt.Sscan(raw, &cfg.grace); err != nil || cfg.grace < 0 {
			return reaperConfig{}, errors.New("config_missing")
		}
	}
	if raw := getenv("INDEX_CLEANUP_BATCH_SIZE"); raw != "" {
		if _, err := fmt.Sscan(raw, &cfg.batch); err != nil || cfg.batch < 1 || cfg.batch > maxBatchSize {
			return reaperConfig{}, errors.New("config_missing")
		}
	}
	return cfg, nil
}

func publicCode(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, errCleanupLost):
		return "cleanup_lost"
	}
	switch err.Error() {
	case "config_missing", "qdrant_failed", "postgres_failed", "cleanup_lost", "cleanup_incomplete":
		return err.Error()
	default:
		return "reaper_failed"
	}
}

func run(ctx context.Context, getenv func(string) string, out io.Writer) error {
	cfg, err := loadConfig(getenv)
	if err != nil {
		return err
	}
	pool, err := pgxpool.New(ctx, cfg.databaseURL)
	if err != nil {
		return errors.New("postgres_failed")
	}
	defer pool.Close()
	budget := time.Duration(cfg.batch)*15*time.Second + time.Minute
	runCtx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	return runReaper(runCtx, cfg, &pgStore{pool: pool}, func(id string) error {
		return deleteQdrantVerified(cfg.qdrantURL, cfg.collection, id)
	}, out)
}

type cleanupClaim struct {
	id       string
	attempts int
	token    uuid.UUID
}

type cleanupStore interface {
	claim(ctx context.Context, graceSeconds, batch int) ([]cleanupClaim, error)
	renew(ctx context.Context, item cleanupClaim) error
	complete(ctx context.Context, item cleanupClaim) error
	mark(ctx context.Context, item cleanupClaim, code string) (string, error)
}

type tally struct {
	claimed   int
	completed int
	retryable int
	terminal  int
	lost      int
}

func runReaper(ctx context.Context, cfg reaperConfig, store cleanupStore, deleteFn func(string) error, out io.Writer) error {
	if cfg.databaseURL == "" || cfg.qdrantURL == "" || cfg.collection == "" {
		return errors.New("config_missing")
	}
	claimed, err := store.claim(ctx, cfg.grace, cfg.batch)
	if err != nil {
		return errors.New("postgres_failed")
	}
	var counts tally
	counts.claimed = len(claimed)
	for _, item := range claimed {
		kind, code := finishOne(ctx, store, deleteFn, item)
		switch kind {
		case "completed":
			counts.completed++
		case "retryable":
			counts.retryable++
			fmt.Fprintf(out, "cleanup_failed id=%s code=%s\n", item.id, code)
		case "terminal":
			counts.terminal++
			fmt.Fprintf(out, "cleanup_failed id=%s code=%s\n", item.id, code)
		default:
			counts.lost++
			fmt.Fprintf(out, "cleanup_lost id=%s\n", item.id)
		}
	}
	fmt.Fprintf(out, "cleanup_claimed=%d\n", counts.claimed)
	fmt.Fprintf(out, "cleanup_completed=%d\n", counts.completed)
	fmt.Fprintf(out, "cleanup_retryable_failed=%d\n", counts.retryable)
	fmt.Fprintf(out, "cleanup_terminal_failed=%d\n", counts.terminal)
	fmt.Fprintf(out, "cleanup_lost=%d\n", counts.lost)
	if counts.retryable > 0 || counts.terminal > 0 || counts.lost > 0 {
		return errors.New("cleanup_incomplete")
	}
	return nil
}

func finishOne(ctx context.Context, store cleanupStore, deleteFn func(string) error, item cleanupClaim) (string, string) {
	if err := store.renew(ctx, item); err != nil {
		return "lost", "cleanup_lost"
	}
	if err := deleteFn(item.id); err != nil {
		return markFailed(ctx, store, item, "qdrant_failed")
	}
	if err := store.complete(ctx, item); err != nil {
		if errors.Is(err, errCleanupLost) {
			return "lost", "cleanup_lost"
		}
		return markFailed(ctx, store, item, "postgres_failed")
	}
	return "completed", ""
}

func markFailed(ctx context.Context, store cleanupStore, item cleanupClaim, code string) (string, string) {
	kind, err := store.mark(ctx, item, code)
	if err != nil && errors.Is(err, errCleanupLost) {
		return "lost", "cleanup_lost"
	}
	if kind != "retryable" && kind != "terminal" {
		return "lost", "cleanup_lost"
	}
	return kind, code
}

type pgStore struct {
	pool *pgxpool.Pool
}

func (s *pgStore) claim(ctx context.Context, graceSeconds, batch int) ([]cleanupClaim, error) {
	token := uuid.New()
	rows, err := s.pool.Query(ctx, claimSQL, graceSeconds, maxCleanupAttempts, token, int(cleanupLease.Seconds()), batch)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []cleanupClaim
	for rows.Next() {
		var item cleanupClaim
		if err := rows.Scan(&item.id, &item.attempts); err != nil {
			return nil, err
		}
		item.token = token
		out = append(out, item)
	}
	return out, rows.Err()
}

func (s *pgStore) renew(ctx context.Context, item cleanupClaim) error {
	tag, err := s.pool.Exec(ctx, renewSQL, item.id, item.token, int(cleanupLease.Seconds()))
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return errCleanupLost
	}
	return nil
}

func (s *pgStore) complete(ctx context.Context, item cleanupClaim) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `DELETE FROM knowledge_chunks WHERE generation_id = $1::uuid`, item.id); err != nil {
		return err
	}
	tag, err := tx.Exec(ctx, `
		UPDATE document_index_generations
		SET cleanup_status = 'COMPLETED',
			cleanup_completed_at = now(),
			cleanup_claim_token = NULL,
			cleanup_claim_expires_at = NULL,
			cleanup_error = NULL,
			updated_at = now()
		WHERE id = $1::uuid AND cleanup_claim_token = $2 AND cleanup_status = 'CLAIMED'`, item.id, item.token)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return errCleanupLost
	}
	return tx.Commit(ctx)
}

func (s *pgStore) mark(ctx context.Context, item cleanupClaim, code string) (string, error) {
	status := "RETRYABLE_FAILED"
	kind := "retryable"
	if item.attempts >= maxCleanupAttempts {
		status = "TERMINAL_FAILED"
		kind = "terminal"
	}
	tag, err := s.pool.Exec(ctx, `
		UPDATE document_index_generations
		SET cleanup_status = $3,
			cleanup_claim_token = NULL,
			cleanup_claim_expires_at = NULL,
			cleanup_error = $4,
			updated_at = now()
		WHERE id = $1::uuid AND cleanup_claim_token = $2 AND cleanup_status = 'CLAIMED'`,
		item.id, item.token, status, code)
	if err != nil {
		return "lost", err
	}
	if tag.RowsAffected() != 1 {
		return "lost", errCleanupLost
	}
	return kind, nil
}

type qdrantBody struct {
	Status string `json:"status"`
	Result *struct {
		Status string `json:"status"`
		Count  *int   `json:"count"`
	} `json:"result"`
}

func deleteQdrantVerified(base, collection, id string) error {
	if base == "" || collection == "" {
		return errors.New("config_missing")
	}
	client := &http.Client{Timeout: 5 * time.Second}
	filter := []byte(`{"filter":{"must":[{"key":"generation_id","match":{"value":"` + id + `"}}]}}`)
	deleted, err := qdrantPost(client, base+"/collections/"+collection+"/points/delete?wait=true", filter)
	if err != nil {
		return errors.New("qdrant_failed")
	}
	if deleted.status != http.StatusOK || !operationCompleted(deleted.payload) {
		return errors.New("qdrant_failed")
	}
	countBody := []byte(`{"filter":{"must":[{"key":"generation_id","match":{"value":"` + id + `"}}]},"exact":true}`)
	counted, err := qdrantPost(client, base+"/collections/"+collection+"/points/count", countBody)
	if err != nil {
		return errors.New("qdrant_failed")
	}
	if counted.status != http.StatusOK || !countIsZero(counted.payload) {
		return errors.New("qdrant_failed")
	}
	return nil
}

type qdrantReply struct {
	status  int
	payload []byte
}

func qdrantPost(client *http.Client, url string, body []byte) (qdrantReply, error) {
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return qdrantReply{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return qdrantReply{}, err
	}
	defer resp.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return qdrantReply{}, err
	}
	return qdrantReply{status: resp.StatusCode, payload: payload}, nil
}

func operationCompleted(payload []byte) bool {
	var body qdrantBody
	if err := json.Unmarshal(payload, &body); err != nil {
		return false
	}
	return body.Status == "ok" && body.Result != nil && body.Result.Status == "completed"
}

func countIsZero(payload []byte) bool {
	var body qdrantBody
	if err := json.Unmarshal(payload, &body); err != nil || body.Status != "ok" || body.Result == nil || body.Result.Count == nil {
		return false
	}
	return *body.Result.Count == 0
}
