package httpserver

import (
	"log/slog"
	"time"

	v1 "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/commune"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/domain"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/procedure"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/session"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/handler"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func New(logger *slog.Logger, pool *pgxpool.Pool, defaultXaID string) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)

	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(middleware.RequestID())
	r.Use(middleware.ApiLog(logger))
	r.Use(middleware.CORS())

	health := &handler.Health{Pool: pool, Timeout: 2 * time.Second}
	r.GET("/health", health.Live)
	r.GET("/ready", health.Ready)
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	communeRepo := &repository.CommuneRepo{Pool: pool}
	communeHandler := commune.NewHandler(communeRepo)
	domainHandler := domain.NewHandler(&repository.DomainRepo{Pool: pool})
	procedureHandler := procedure.NewHandler(&repository.ProcedureRepo{Pool: pool}, defaultXaID)
	sessionHandler := session.NewHandler(&repository.SessionRepo{Pool: pool}, communeRepo)

	api := r.Group("/api")
	v1.MapRoutes(api, communeHandler, domainHandler, procedureHandler, sessionHandler)

	return r
}
