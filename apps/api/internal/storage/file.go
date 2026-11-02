package storage

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/sys/unix"
)

// ErrUnsafePath is returned when a key breaks the key contract or a path component
// under the root is a symlink or not a directory.
var ErrUnsafePath = errors.New("storage: unsafe object path")

const (
	maxKeyBytes      = 512
	maxKeyComponents = 8
	maxComponentLen  = 128
	tempPrefix       = ".tmp-"
)

// File stores objects as regular files at <root>/<bucket>/<key> (contract:
// deploy/contracts/object_store_keys.json, shared with the ai-service reader).
//
// Every path component is opened relative to its parent directory with
// O_NOFOLLOW, starting from a descriptor of the root itself, so neither a ".."
// key nor a symlink placed anywhere below the root can lead a read, write or
// delete outside it. Writes go to a temporary file in the target directory,
// are fsynced, then renamed into place, and the directory is fsynced: a reader
// sees the whole object or none of it.
type File struct {
	root   string
	bucket string
}

// NewFile checks that root is an absolute, real directory (not a symlink) and
// that bucket is a valid component. It does not create anything.
func NewFile(root, bucket string) (*File, error) {
	if !filepath.IsAbs(root) || filepath.Clean(root) != root {
		return nil, fmt.Errorf("object storage file root must be an absolute clean path")
	}
	if err := validComponent(bucket); err != nil {
		return nil, fmt.Errorf("object storage bucket is invalid")
	}
	st, err := os.Lstat(root)
	if err != nil || !st.IsDir() || st.Mode()&fs.ModeSymlink != 0 {
		return nil, fmt.Errorf("object storage file root is not a directory")
	}
	return &File{root: root, bucket: bucket}, nil
}

// ValidateKey applies the shared key contract.
func ValidateKey(key string) error {
	if key == "" || len(key) > maxKeyBytes {
		return ErrUnsafePath
	}
	parts := strings.Split(key, "/")
	if len(parts) > maxKeyComponents {
		return ErrUnsafePath
	}
	for _, p := range parts {
		if err := validComponent(p); err != nil {
			return err
		}
	}
	return nil
}

func validComponent(c string) error {
	if c == "" || len(c) > maxComponentLen || c[0] == '.' || c[0] == '-' {
		return ErrUnsafePath
	}
	for i := 0; i < len(c); i++ {
		b := c[i]
		switch {
		case b >= 'a' && b <= 'z', b >= 'A' && b <= 'Z', b >= '0' && b <= '9', b == '.', b == '_', b == '-':
		default:
			return ErrUnsafePath
		}
	}
	return nil
}

// openDirs walks <root>/<bucket>/<dirs...> one component at a time. With create,
// missing directories are made (0755). It returns a descriptor of the last one.
func (f *File) openDirs(dirs []string, create bool) (int, error) {
	fd, err := unix.Open(f.root, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return -1, mapErr(err)
	}
	for _, c := range append([]string{f.bucket}, dirs...) {
		if create {
			if err := unix.Mkdirat(fd, c, 0o755); err != nil && !errors.Is(err, unix.EEXIST) {
				unix.Close(fd)
				return -1, mapErr(err)
			}
		}
		next, err := unix.Openat(fd, c, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
		unix.Close(fd)
		if err != nil {
			return -1, mapErr(err)
		}
		fd = next
	}
	return fd, nil
}

func mapErr(err error) error {
	switch {
	case errors.Is(err, unix.ENOENT):
		return ErrNotFound
	case errors.Is(err, unix.ELOOP), errors.Is(err, unix.ENOTDIR), errors.Is(err, unix.EMLINK):
		return ErrUnsafePath
	default:
		return fmt.Errorf("object storage file operation failed")
	}
}

func split(key string) ([]string, string, error) {
	if err := ValidateKey(key); err != nil {
		return nil, "", err
	}
	parts := strings.Split(key, "/")
	return parts[:len(parts)-1], parts[len(parts)-1], nil
}

// EnsureBucket requires <root>/<bucket> to exist as a real directory. The
// deployment creates it; this store never creates its own bucket.
func (f *File) EnsureBucket(context.Context) error {
	fd, err := f.openDirs(nil, false)
	if err != nil {
		return fmt.Errorf("object storage bucket directory is not ready")
	}
	return unix.Close(fd)
}

// Put writes exactly size bytes from r (size < 0: until EOF) atomically.
func (f *File) Put(ctx context.Context, key string, r io.Reader, size int64, _ string) error {
	dirs, name, err := split(key)
	if err != nil {
		return err
	}
	dfd, err := f.openDirs(dirs, true)
	if err != nil {
		return err
	}
	defer unix.Close(dfd)
	var rnd [8]byte
	if _, err := rand.Read(rnd[:]); err != nil {
		return fmt.Errorf("object storage put failed")
	}
	tmp := tempPrefix + hex.EncodeToString(rnd[:])
	tfd, err := unix.Openat(dfd, tmp, unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0o640)
	if err != nil {
		return mapErr(err)
	}
	out := os.NewFile(uintptr(tfd), tmp)
	committed := false
	defer func() {
		if !committed {
			out.Close()
			_ = unix.Unlinkat(dfd, tmp, 0)
		}
	}()
	src := r
	if size >= 0 {
		src = io.LimitReader(r, size+1)
	}
	n, err := io.Copy(out, &ctxReader{ctx: ctx, r: src})
	if err != nil {
		return fmt.Errorf("object storage put failed")
	}
	if size >= 0 && n != size {
		return fmt.Errorf("object storage put failed: size mismatch")
	}
	if err := out.Sync(); err != nil {
		return fmt.Errorf("object storage put failed")
	}
	if err := out.Close(); err != nil {
		return fmt.Errorf("object storage put failed")
	}
	// rename(2) replaces a directory entry and never follows a symlink at the target.
	if err := unix.Renameat(dfd, tmp, dfd, name); err != nil {
		return mapErr(err)
	}
	committed = true
	if err := unix.Fsync(dfd); err != nil {
		return fmt.Errorf("object storage put failed: directory not synced")
	}
	return nil
}

// Get opens the object for reading. The caller closes it.
func (f *File) Get(_ context.Context, key string) (io.ReadCloser, error) {
	dirs, name, err := split(key)
	if err != nil {
		return nil, err
	}
	dfd, err := f.openDirs(dirs, false)
	if err != nil {
		return nil, err
	}
	defer unix.Close(dfd)
	// O_NONBLOCK: opening a FIFO planted at the object path must not block; blocking
	// mode is restored once the descriptor is known to be a regular file.
	fd, err := unix.Openat(dfd, name, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_NONBLOCK|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, mapErr(err)
	}
	var st unix.Stat_t
	if err := unix.Fstat(fd, &st); err != nil || st.Mode&unix.S_IFMT != unix.S_IFREG {
		unix.Close(fd)
		return nil, ErrUnsafePath
	}
	if err := unix.SetNonblock(fd, false); err != nil {
		unix.Close(fd)
		return nil, fmt.Errorf("object storage get failed")
	}
	return os.NewFile(uintptr(fd), name), nil
}

// Delete removes the object. A missing object is not an error.
func (f *File) Delete(_ context.Context, key string) error {
	dirs, name, err := split(key)
	if err != nil {
		return err
	}
	dfd, err := f.openDirs(dirs, false)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	defer unix.Close(dfd)
	if err := unix.Unlinkat(dfd, name, 0); err != nil && !errors.Is(err, unix.ENOENT) {
		return mapErr(err)
	}
	if err := unix.Fsync(dfd); err != nil {
		return fmt.Errorf("object storage delete failed: directory not synced")
	}
	return nil
}

// PresignGet is not supported: objects are only served through the API.
func (f *File) PresignGet(context.Context, string, time.Duration) (string, error) {
	return "", fmt.Errorf("object storage presign is not supported by the file store")
}

// CleanupTemp removes temporary files older than maxAge left by interrupted puts.
// It walks without following symlinks and returns how many it removed.
func (f *File) CleanupTemp(maxAge time.Duration) (int, error) {
	base := filepath.Join(f.root, f.bucket)
	cutoff := time.Now().Add(-maxAge)
	removed := 0
	err := filepath.WalkDir(base, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.Type()&fs.ModeSymlink != 0 || !strings.HasPrefix(d.Name(), tempPrefix) || d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil || !info.Mode().IsRegular() || info.ModTime().After(cutoff) {
			return nil
		}
		if os.Remove(path) == nil {
			removed++
		}
		return nil
	})
	return removed, err
}

type ctxReader struct {
	ctx context.Context
	r   io.Reader
}

func (c *ctxReader) Read(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.Read(p)
}
