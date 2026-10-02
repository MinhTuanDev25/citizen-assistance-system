package httpserver

import (
	"log/slog"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/aiextract"
	v1 "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1"
	authapi "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/commune"
	documentapi "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/document"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/domain"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/procedure"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/session"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/chat"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/document"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/handler"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/storage"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func New(logger *slog.Logger, pool *pgxpool.Pool, cfg config.Config) *gin.Engine {
	return NewWithStore(logger, pool, cfg, nil)
}

// NewWithStore is New, with an object store supplied by the caller.
// Ingestion routes are mounted only when the flag is on and objects is non-nil.
// Keyword-only startup passes a nil store and does not contact object storage.
func NewWithStore(logger *slog.Logger, pool *pgxpool.Pool, cfg config.Config, objects storage.ObjectStore) *gin.Engine {
	return NewWithIndexWorker(logger, pool, cfg, objects, nil)
}

// NewWithIndexWorker mounts indexing when cfg.AdminIndexingEnabled.
// A nil worker uses the HTTP mock client. Tests pass an in-process worker.
func NewWithIndexWorker(logger *slog.Logger, pool *pgxpool.Pool, cfg config.Config, objects storage.ObjectStore, worker index.Worker) *gin.Engine {
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

	tokens := &auth.TokenService{
		Secret: []byte(cfg.JWTSecret),
		TTL:    time.Duration(cfg.JWTExpireHours) * time.Hour,
	}

	communeRepo := &repository.CommuneRepo{Pool: pool}
	sessionRepo := &repository.SessionRepo{Pool: pool}
	userRepo := &repository.UserRepo{Pool: pool}
	communeHandler := commune.NewHandler(communeRepo)
	domainHandler := domain.NewHandler(&repository.DomainRepo{Pool: pool})
	procedureHandler := procedure.NewHandler(&repository.ProcedureRepo{Pool: pool}, cfg.XAID, cfg.CitizenDomainIDs)

	// AI extraction (P2) is opt-in and off by default. An empty/disabled
	// config yields a nil Extractor, so chat.Service.Turn runs the exact
	// pre-P2 keyword-only path — see chat.Service.aiEnabled().
	var extractor aiextract.Extractor
	if cfg.AIExtractEnabled && cfg.AIServiceURL != "" {
		client := aiextract.NewClient(cfg.AIServiceURL, time.Duration(cfg.AIExtractTimeoutMS)*time.Millisecond)
		client.ServiceToken = cfg.AIServiceToken
		extractor = client
	}
	sessionHandler := session.NewHandler(sessionRepo, communeRepo, &chat.Service{
		Pool:                pool,
		Sessions:            sessionRepo,
		CitizenDomains:      append([]string{}, cfg.CitizenDomainIDs...),
		Extractor:           extractor,
		AIEnabled:           cfg.AIExtractEnabled && extractor != nil,
		AIPolicy:            decision.Policy{SelectMin: cfg.AIIntentSelectMin, ConfirmMin: cfg.AIIntentConfirmMin},
		AIPolicySet:         true,
		AISlotConfidenceMin: cfg.AISlotConfidenceMin,
	})
	authHandler := authapi.NewHandler(userRepo, tokens)

	var documentHandler *documentapi.Handler
	if cfg.AdminIngestionEnabled && objects != nil {
		handler := documentapi.NewHandler(&document.Service{
			Repo:     &repository.DocumentRepo{Pool: pool},
			Objects:  objects,
			Bucket:   cfg.ObjectStorageBucket,
			XAID:     cfg.XAID,
			MaxBytes: cfg.DocumentMaxBytes,
			Logger:   logger,
		})
		if cfg.AdminIndexingEnabled {
			if worker == nil {
				worker = index.HTTPWorker{BaseURL: cfg.AIServiceURL, Token: cfg.AIServiceToken, Timeout: workerTimeout(cfg)}
			}
			handler.Index = &index.Service{
				XAID: cfg.XAID, Store: &repository.IndexRepo{Pool: pool}, Worker: worker,
				Timeout: workerTimeout(cfg), ClaimTTL: cfg.IndexClaimTTL,
				Pipeline: cfg.IndexMode == "pipeline", Bucket: cfg.ObjectStorageBucket,
				PublishTimeout: cfg.IndexPublishTimeout,
			}
		}
		documentHandler = handler
	}

	api := r.Group("/api")
	v1.MapRoutes(
		api,
		communeHandler,
		domainHandler,
		procedureHandler,
		sessionHandler,
		authHandler,
		documentHandler,
		middleware.OptionalJWT(tokens),
		middleware.RequireJWT(tokens),
		middleware.RequireAdmin(tokens),
	)

	return r
}

func workerTimeout(cfg config.Config) time.Duration {
	if cfg.IndexMode == "pipeline" {
		return cfg.IndexPipelineTimeout
	}
	return cfg.IndexTimeout
}
