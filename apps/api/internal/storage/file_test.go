package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

type keyContract struct {
	Valid          []string `json:"valid"`
	Invalid        []string `json:"invalid"`
	ValidBuckets   []string `json:"valid_buckets"`
	InvalidBuckets []string `json:"invalid_buckets"`
}

func loadContract(t *testing.T) keyContract {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "deploy", "contracts", "object_store_keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c keyContract
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return c
}

func newStore(t *testing.T) (*File, string) {
	t.Helper()
	root := t.TempDir()
	root, _ = filepath.EvalSymlinks(root) // macOS: /var -> /private/var
	if err := os.Mkdir(filepath.Join(root, "cas-documents"), 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := NewFile(root, "cas-documents")
	if err != nil {
		t.Fatal(err)
	}
	return f, root
}

func TestFileKeyContract(t *testing.T) {
	c := loadContract(t)
	for _, k := range c.Valid {
		if err := ValidateKey(k); err != nil {
			t.Errorf("valid key refused: %q", k)
		}
	}
	for _, k := range c.Invalid {
		if err := ValidateKey(k); !errors.Is(err, ErrUnsafePath) {
			t.Errorf("invalid key accepted: %q", k)
		}
	}
	if ValidateKey(strings.Repeat("a", 129)) == nil || ValidateKey(strings.Repeat("a/", 300)+"a") == nil {
		t.Error("over-long component or key accepted")
	}
	if ValidateKey(strings.Repeat("a", 128)) != nil {
		t.Error("128-character component refused")
	}
	root := t.TempDir()
	for _, b := range c.ValidBuckets {
		if _, err := NewFile(root, b); err != nil {
			t.Errorf("valid bucket refused: %q", b)
		}
	}
	for _, b := range c.InvalidBuckets {
		if _, err := NewFile(root, b); err == nil {
			t.Errorf("invalid bucket accepted: %q", b)
		}
	}
}

func TestFilePutGetDeleteRoundTrip(t *testing.T) {
	f, root := newStore(t)
	ctx := context.Background()
	key := "xa_chu_se/documents/d1/abc.pdf"
	body := []byte("%PDF-1.4 test body")
	if err := f.Put(ctx, key, bytes.NewReader(body), int64(len(body)), "application/pdf"); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(filepath.Join(root, "cas-documents", key)); !bytes.Equal(got, body) {
		t.Fatalf("file on disk = %q", got)
	}
	rc, err := f.Get(ctx, key)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(rc)
	rc.Close()
	if !bytes.Equal(got, body) {
		t.Fatalf("Get = %q", got)
	}
	if err := f.Delete(ctx, key); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Get(ctx, key); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Get after delete = %v", err)
	}
	if err := f.Delete(ctx, key); err != nil {
		t.Fatalf("second delete = %v", err)
	}
	if err := f.Delete(ctx, "xa/missing/dir/x.pdf"); err != nil {
		t.Fatalf("delete under a missing directory = %v", err)
	}
	assertNoTemp(t, root)
}

func TestFilePutSizeMismatchLeavesNothing(t *testing.T) {
	f, root := newStore(t)
	key := "xa/documents/d1/short.pdf"
	if err := f.Put(context.Background(), key, strings.NewReader("12345"), 9, ""); err == nil {
		t.Fatal("short body accepted")
	}
	if err := f.Put(context.Background(), key, strings.NewReader("123456789"), 4, ""); err == nil {
		t.Fatal("long body accepted")
	}
	if _, err := os.Lstat(filepath.Join(root, "cas-documents", key)); !os.IsNotExist(err) {
		t.Fatal("partial object published")
	}
	assertNoTemp(t, root)
}

func TestFilePutCancelledLeavesNothing(t *testing.T) {
	f, root := newStore(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := f.Put(ctx, "xa/doc.pdf", strings.NewReader("x"), 1, ""); err == nil {
		t.Fatal("cancelled put succeeded")
	}
	if _, err := os.Lstat(filepath.Join(root, "cas-documents", "xa", "doc.pdf")); !os.IsNotExist(err) {
		t.Fatal("object published after cancel")
	}
	assertNoTemp(t, root)
}

func TestFileRefusesSymlinkEscapes(t *testing.T) {
	f, root := newStore(t)
	outside := t.TempDir()
	secret := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(secret, []byte("research data"), 0o644); err != nil {
		t.Fatal(err)
	}
	bucket := filepath.Join(root, "cas-documents")
	ctx := context.Background()

	// A directory component that is a symlink to outside the root.
	if err := os.Symlink(outside, filepath.Join(bucket, "xa")); err != nil {
		t.Fatal(err)
	}
	if err := f.Put(ctx, "xa/new.pdf", strings.NewReader("x"), 1, ""); !errors.Is(err, ErrUnsafePath) {
		t.Fatalf("put through a symlinked directory = %v", err)
	}
	if _, err := os.Lstat(filepath.Join(outside, "new.pdf")); !os.IsNotExist(err) {
		t.Fatal("put wrote outside the root")
	}
	if _, err := f.Get(ctx, "xa/secret.txt"); !errors.Is(err, ErrUnsafePath) {
		t.Fatalf("get through a symlinked directory = %v", err)
	}
	if err := f.Delete(ctx, "xa/secret.txt"); !errors.Is(err, ErrUnsafePath) {
		t.Fatalf("delete through a symlinked directory = %v", err)
	}
	if _, err := os.Stat(secret); err != nil {
		t.Fatal("delete removed a file outside the root")
	}

	// The final component is a symlink to a file outside the root.
	if err := os.MkdirAll(filepath.Join(bucket, "real"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(bucket, "real", "link.pdf")); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Get(ctx, "real/link.pdf"); !errors.Is(err, ErrUnsafePath) {
		t.Fatalf("get of a symlinked file = %v", err)
	}
	// Put over a symlink replaces the link itself, never writes through it.
	if err := f.Put(ctx, "real/link.pdf", strings.NewReader("new"), 3, ""); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(secret); string(got) != "research data" {
		t.Fatal("put wrote through a symlink")
	}
	if st, _ := os.Lstat(filepath.Join(bucket, "real", "link.pdf")); st.Mode()&os.ModeSymlink != 0 {
		t.Fatal("symlink still in place after put")
	}

	// The bucket itself is a symlink.
	root2 := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(root2, "cas-documents")); err != nil {
		t.Fatal(err)
	}
	f2, err := NewFile(root2, "cas-documents")
	if err != nil {
		t.Fatal(err)
	}
	if err := f2.EnsureBucket(ctx); err == nil {
		t.Fatal("symlinked bucket accepted")
	}
	if err := f2.Put(ctx, "a.pdf", strings.NewReader("x"), 1, ""); !errors.Is(err, ErrUnsafePath) {
		t.Fatalf("put into a symlinked bucket = %v", err)
	}

	// The root is a symlink.
	link := filepath.Join(t.TempDir(), "root-link")
	if err := os.Symlink(root, link); err != nil {
		t.Fatal(err)
	}
	if _, err := NewFile(link, "cas-documents"); err == nil {
		t.Fatal("symlinked root accepted")
	}
	// Traversal keys never reach the filesystem.
	for _, k := range []string{"../x.pdf", "xa/../../x.pdf", "/etc/passwd"} {
		if err := f.Put(ctx, k, strings.NewReader("x"), 1, ""); !errors.Is(err, ErrUnsafePath) {
			t.Fatalf("put %q = %v", k, err)
		}
	}
}

func TestFileEnsureBucketRequiresTheDirectory(t *testing.T) {
	root := t.TempDir()
	f, err := NewFile(root, "cas-documents")
	if err != nil {
		t.Fatal(err)
	}
	if err := f.EnsureBucket(context.Background()); err == nil {
		t.Fatal("missing bucket directory accepted")
	}
	if _, err := os.Lstat(filepath.Join(root, "cas-documents")); !os.IsNotExist(err) {
		t.Fatal("EnsureBucket created the bucket")
	}
}

func TestFileConcurrentReplaceIsAtomic(t *testing.T) {
	f, _ := newStore(t)
	ctx := context.Background()
	key := "xa/documents/d/obj.pdf"
	a, b := bytes.Repeat([]byte("A"), 64<<10), bytes.Repeat([]byte("B"), 64<<10)
	if err := f.Put(ctx, key, bytes.NewReader(a), int64(len(a)), ""); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	stop := make(chan struct{})
	bad := make(chan string, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			rc, err := f.Get(ctx, key)
			if err != nil {
				bad <- "get: " + err.Error()
				return
			}
			got, _ := io.ReadAll(rc)
			rc.Close()
			if !bytes.Equal(got, a) && !bytes.Equal(got, b) {
				bad <- "partial object read"
				return
			}
		}
	}()
	for i := 0; i < 100; i++ {
		body := a
		if i%2 == 0 {
			body = b
		}
		if err := f.Put(ctx, key, bytes.NewReader(body), int64(len(body)), ""); err != nil {
			t.Fatal(err)
		}
	}
	close(stop)
	wg.Wait()
	select {
	case msg := <-bad:
		t.Fatal(msg)
	default:
	}
}

func TestFileCleanupTemp(t *testing.T) {
	f, root := newStore(t)
	dir := filepath.Join(root, "cas-documents", "xa")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	old, fresh, keep := filepath.Join(dir, ".tmp-old"), filepath.Join(dir, ".tmp-fresh"), filepath.Join(dir, "doc.pdf")
	for _, p := range []string{old, fresh, keep} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	past := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(old, past, past); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	outsideTemp := filepath.Join(outside, ".tmp-outside")
	if err := os.WriteFile(outsideTemp, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(outsideTemp, past, past); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "cas-documents", "link")); err != nil {
		t.Fatal(err)
	}
	n, err := f.CleanupTemp(time.Hour)
	if err != nil || n != 1 {
		t.Fatalf("CleanupTemp = %d, %v", n, err)
	}
	for _, p := range []string{fresh, keep, outsideTemp} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("%s was removed", p)
		}
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatal("old temporary file kept")
	}
}

func TestFilePresignIsRefused(t *testing.T) {
	f, _ := newStore(t)
	if _, err := f.PresignGet(context.Background(), "a.pdf", time.Minute); err == nil {
		t.Fatal("presign succeeded")
	}
}

func assertNoTemp(t *testing.T, root string) {
	t.Helper()
	_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err == nil && strings.HasPrefix(d.Name(), tempPrefix) {
			t.Errorf("temporary file left: %s", p)
		}
		return nil
	})
}

func TestFileGetRefusesFIFOWithoutBlocking(t *testing.T) {
	f, root := newStore(t)
	dir := filepath.Join(root, "cas-documents", "xa")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := unix.Mkfifo(filepath.Join(dir, "pipe.pdf"), 0o644); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		rc, err := f.Get(context.Background(), "xa/pipe.pdf")
		if rc != nil {
			rc.Close()
		}
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, ErrUnsafePath) {
			t.Fatalf("Get on a FIFO: %v, want ErrUnsafePath", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Get blocked on a FIFO")
	}
}

func TestFileGetReturnsABlockingReader(t *testing.T) {
	f, _ := newStore(t)
	if err := f.Put(context.Background(), "xa/doc.pdf", strings.NewReader("hello"), 5, ""); err != nil {
		t.Fatal(err)
	}
	rc, err := f.Get(context.Background(), "xa/doc.pdf")
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()
	fl, err := unix.FcntlInt(rc.(*os.File).Fd(), unix.F_GETFL, 0)
	if err != nil || fl&unix.O_NONBLOCK != 0 {
		t.Fatalf("reader flags %#x err %v: O_NONBLOCK still set", fl, err)
	}
	b, _ := io.ReadAll(rc)
	if string(b) != "hello" {
		t.Fatalf("read %q", b)
	}
}
