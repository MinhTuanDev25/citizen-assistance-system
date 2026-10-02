// Command bootstrap-admin sets/creates the production admin password from env.
// Fail-closed: requires BOOTSTRAP_ADMIN_EMAIL + BOOTSTRAP_ADMIN_PASSWORD (>= 12 chars).
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/db"
)

func main() {
	email := strings.TrimSpace(os.Getenv("BOOTSTRAP_ADMIN_EMAIL"))
	password := os.Getenv("BOOTSTRAP_ADMIN_PASSWORD")
	if email == "" || password == "" {
		fmt.Fprintln(os.Stderr, "bootstrap-admin refused: set BOOTSTRAP_ADMIN_EMAIL and BOOTSTRAP_ADMIN_PASSWORD")
		os.Exit(2)
	}
	if len(password) < 12 {
		fmt.Fprintln(os.Stderr, "bootstrap-admin refused: BOOTSTRAP_ADMIN_PASSWORD must be at least 12 characters")
		os.Exit(2)
	}
	if strings.EqualFold(password, "admin123") || strings.EqualFold(password, "citizen123") {
		fmt.Fprintln(os.Stderr, "bootstrap-admin refused: demo passwords are not allowed")
		os.Exit(2)
	}

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	pool, err := db.NewPool(ctx, cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer pool.Close()

	hash, err := auth.HashPassword(password)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	name := strings.TrimSpace(os.Getenv("BOOTSTRAP_ADMIN_NAME"))
	if name == "" {
		name = "Administrator"
	}

	tag, err := pool.Exec(ctx, `
		UPDATE users
		SET password_hash = $2, role = 'ADMIN', full_name = COALESCE(NULLIF(full_name,''), $3), updated_at = now()
		WHERE lower(email) = lower($1)`, email, hash, name)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if tag.RowsAffected() == 0 {
		_, err = pool.Exec(ctx, `
			INSERT INTO users (role, full_name, email, password_hash)
			VALUES ('ADMIN', $1, $2, $3)`, name, email, hash)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Println("bootstrap-admin: created admin", email)
		return
	}
	fmt.Println("bootstrap-admin: updated admin", email)
}
