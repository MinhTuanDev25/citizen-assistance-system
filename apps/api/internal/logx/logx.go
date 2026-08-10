package logx

import (
	"log/slog"
	"os"
	"strings"
)

// New returns a JSON structured logger for stdout (Promtail/Loki scrapes this).
func New(level, env string) *slog.Logger {
	var lvl slog.Level
	switch strings.ToLower(level) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn", "warning":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}

	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lvl})
	return slog.New(handler).With(
		"service", "cas-api",
		"env", env,
	)
}
