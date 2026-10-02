//go:build miniopersist

package storage

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestMinIOVolumeSurvivesContainerRecreate(t *testing.T) {
	if _, err := exec.LookPath("docker"); err != nil {
		t.Fatal(err)
	}
	root := filepath.Clean(filepath.Join("..", "..", "..", "..", "deploy"))
	project := "casp31persist"
	payload := []byte("%PDF-1.4 persist-bytes")
	sum := sha256.Sum256(payload)
	want := hex.EncodeToString(sum[:])
	env := append(os.Environ(),
		"COMPOSE_PROJECT_NAME="+project,
		"MINIO_ROOT_USER=minioadmin",
		"MINIO_ROOT_PASSWORD=minioadmin",
		"MINIO_API_PORT=19090",
		"MINIO_CONSOLE_PORT=19091",
	)
	t.Cleanup(func() {
		cmd := exec.Command("docker", "compose", "--profile", "storage", "down", "-v", "--remove-orphans")
		cmd.Dir = root
		cmd.Env = env
		_ = cmd.Run()
	})
	compose(t, root, env, "up", "-d", "minio")
	container := project + "-minio-1"
	waitMinIO(t, container)
	dest := inspectMount(t, container)
	if !strings.Contains(" "+dest+" ", " /bitnami/minio/data ") {
		t.Fatalf("volume destination %q", dest)
	}
	for _, mount := range strings.Fields(dest) {
		if mount == "/data" {
			t.Fatalf("named volume mounted at /data: %q", dest)
		}
	}
	mc(t, container, "alias", "set", "local", "http://127.0.0.1:9000", "minioadmin", "minioadmin")
	mc(t, container, "mb", "--ignore-existing", "local/cas-documents")
	pipe := exec.Command("docker", "exec", "-i", container, "/opt/bitnami/minio-client/bin/mc", "pipe", "local/cas-documents/persist.pdf")
	pipe.Stdin = bytes.NewReader(payload)
	if out, err := pipe.CombinedOutput(); err != nil {
		t.Fatalf("pipe: %v %s", err, out)
	}
	compose(t, root, env, "stop", "minio")
	compose(t, root, env, "rm", "-f", "minio")
	compose(t, root, env, "up", "-d", "minio")
	waitMinIO(t, container)
	mc(t, container, "alias", "set", "local", "http://127.0.0.1:9000", "minioadmin", "minioadmin")
	got := mcOut(t, container, "cat", "local/cas-documents/persist.pdf")
	if !bytes.Equal(got, payload) {
		t.Fatalf("bytes mismatch after recreate: %q", got)
	}
	sum2 := sha256.Sum256(got)
	if hex.EncodeToString(sum2[:]) != want {
		t.Fatalf("sha mismatch")
	}
}

func compose(t *testing.T, dir string, env []string, args ...string) {
	t.Helper()
	cmd := exec.Command("docker", append([]string{"compose", "--profile", "storage"}, args...)...)
	cmd.Dir = dir
	cmd.Env = env
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compose %v: %v %s", args, err, out)
	}
}

func waitMinIO(t *testing.T, container string) {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	var last []byte
	for time.Now().Before(deadline) {
		cmd := exec.Command("docker", "exec", container, "curl", "-sf", "http://127.0.0.1:9000/minio/health/live")
		out, err := cmd.CombinedOutput()
		last = out
		if err == nil {
			return
		}
		time.Sleep(time.Second)
	}
	t.Fatalf("minio not live: %s", last)
}

func inspectMount(t *testing.T, container string) string {
	t.Helper()
	cmd := exec.Command("docker", "inspect", "-f", "{{range .Mounts}}{{.Destination}} {{end}}", container)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("inspect: %v %s", err, out)
	}
	return string(bytes.TrimSpace(out))
}

func mc(t *testing.T, container string, args ...string) {
	t.Helper()
	if out := mcOut(t, container, args...); len(out) > 0 && bytes.Contains(out, []byte("ERROR")) {
		t.Fatalf("mc %v: %s", args, out)
	}
}

func mcOut(t *testing.T, container string, args ...string) []byte {
	t.Helper()
	cmd := exec.Command("docker", append([]string{"exec", container, "/opt/bitnami/minio-client/bin/mc"}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("mc %v: %v %s", args, err, out)
	}
	return out
}
