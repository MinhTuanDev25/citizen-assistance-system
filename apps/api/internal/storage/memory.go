package storage

import (
	"bytes"
	"context"
	"io"
	"sync"
	"time"
)

// Memory is an in-process store for tests. Put buffers the object because
// tests use small PDFs; the production MinIO adapter streams.
type Memory struct {
	mu      sync.Mutex
	objects map[string][]byte
	FailPut bool
	puts    int
	deletes int
}

func NewMemory() *Memory {
	return &Memory{objects: map[string][]byte{}}
}

func (m *Memory) EnsureBucket(context.Context) error { return nil }

func (m *Memory) Put(_ context.Context, key string, r io.Reader, _ int64, _ string) error {
	if m.FailPut {
		return errorsPut
	}
	buf, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.objects[key] = buf
	m.puts++
	return nil
}

func (m *Memory) Get(_ context.Context, key string) (io.ReadCloser, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	b, ok := m.objects[key]
	if !ok {
		return nil, ErrNotFound
	}
	return io.NopCloser(bytes.NewReader(b)), nil
}

func (m *Memory) Delete(_ context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.objects, key)
	m.deletes++
	return nil
}

func (m *Memory) PresignGet(_ context.Context, key string, _ time.Duration) (string, error) {
	return "memory://" + key, nil
}

func (m *Memory) Len() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.objects)
}

func (m *Memory) Puts() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.puts
}

func (m *Memory) Deletes() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.deletes
}

var errorsPut = errPut("storage: put failed")

type errPut string

func (e errPut) Error() string { return string(e) }
