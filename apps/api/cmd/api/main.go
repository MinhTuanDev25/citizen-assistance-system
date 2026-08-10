package main

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	_ "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/docs"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/db"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/httpserver"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/logx"
)

// @title           Citizen Assistance API
// @version         1.0
// @description     Backend API for commune citizen assistance (Chư Sê).
// @BasePath        /
// @schemes         http
// @accept          json
// @produce         json
// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
// @description Type "Bearer" followed by a space and JWT.
func main() {
	cfg, err := config.Load()
	if err != nil {
		panic(err)
	}

	logger := logx.New(cfg.LogLevel, cfg.Env)
	logger.Info("starting api",
		"env", cfg.Env,
		"config_file", cfg.ConfigFile,
		"addr", cfg.APIAddr,
		"xa_id", cfg.XAID,
	)

	ctx := context.Background()
	pool, err := db.NewPool(ctx, cfg)
	if err != nil {
		logger.Error("database connection failed", "error", err)
		os.Exit(1)
	}
	defer pool.Close()
	logger.Info("database connected",
		"max_conns", cfg.DBMaxConns,
		"min_conns", cfg.DBMinConns,
	)

	engine := httpserver.New(logger, pool, cfg)
	srv := &http.Server{
		Addr:              cfg.APIAddr,
		Handler:           engine,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		logger.Info("http listen", "addr", cfg.APIAddr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("http server error", "error", err)
			os.Exit(1)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	logger.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("shutdown error", "error", err)
	}
}
