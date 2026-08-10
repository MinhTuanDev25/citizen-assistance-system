package httpserver

import (
	"log/slog"
	"time"

	v1 "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/commune"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/domain"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/handler"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
)

func New(logger *slog.Logger, pool *pgxpool.Pool) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)

	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(middleware.RequestID())
	r.Use(middleware.ApiLog(logger))
	r.Use(middleware.CORS())

	health := &handler.Health{Pool: pool, Timeout: 2 * time.Second}
	r.GET("/health", health.Live)
	r.GET("/ready", health.Ready)

	communeHandler := commune.NewHandler(&repository.CommuneRepo{Pool: pool})
	domainHandler := domain.NewHandler(&repository.DomainRepo{Pool: pool})

	api := r.Group("/api")
	v1.MapRoutes(api, communeHandler, domainHandler)

	return r
}
