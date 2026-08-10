package authapi

import (
	"errors"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgconn"
)

type Handler struct {
	Users  *repository.UserRepo
	Tokens *auth.TokenService
}

func NewHandler(users *repository.UserRepo, tokens *auth.TokenService) *Handler {
	return &Handler{Users: users, Tokens: tokens}
}

type registerRequest struct {
	FullName string `json:"full_name" binding:"required"`
	Email    string `json:"email" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type loginRequest struct {
	Email    string `json:"email" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// Register godoc
//
//	@Summary		Register citizen account
//	@Tags			auth
//	@Accept			json
//	@Produce		json
//	@Param			body	body		registerRequest	true	"Citizen registration"
//	@Success		201		{object}	response.Envelope
//	@Failure		400		{object}	response.Envelope
//	@Failure		409		{object}	response.Envelope
//	@Failure		500		{object}	response.Envelope
//	@Router			/api/v1/auth/register [post]
func (h *Handler) Register(c *gin.Context) {
	var req registerRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "full_name, email, password are required")
		return
	}
	fullName := strings.TrimSpace(req.FullName)
	email := strings.TrimSpace(strings.ToLower(req.Email))
	password := req.Password
	if fullName == "" || email == "" || password == "" {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "full_name, email, password are required")
		return
	}
	if utf8.RuneCountInString(fullName) < 2 {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "full_name is too short")
		return
	}
	if !strings.Contains(email, "@") {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "email is invalid")
		return
	}
	if len(password) < 6 {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "password must be at least 6 characters")
		return
	}

	hash, err := auth.HashPassword(password)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to hash password", err)
		return
	}

	user, err := h.Users.CreateCitizen(c.Request.Context(), fullName, email, hash)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			response.Fail(c, http.StatusConflict, "CONFLICT", "email already registered")
			return
		}
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to create user", err)
		return
	}

	token, expiresAt, err := h.issueToken(user)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to issue token", err)
		return
	}

	response.Created(c, gin.H{
		"access_token": token,
		"token_type":   "Bearer",
		"expires_at":   expiresAt.UTC().Format(time.RFC3339),
		"user":         user.Public(),
	})
}

// Login godoc
//
//	@Summary		Login (citizen or admin)
//	@Tags			auth
//	@Accept			json
//	@Produce		json
//	@Param			body	body		loginRequest	true	"Credentials"
//	@Success		200		{object}	response.Envelope
//	@Failure		400		{object}	response.Envelope
//	@Failure		401		{object}	response.Envelope
//	@Failure		500		{object}	response.Envelope
//	@Router			/api/v1/auth/login [post]
func (h *Handler) Login(c *gin.Context) {
	var req loginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "email and password are required")
		return
	}
	email := strings.TrimSpace(strings.ToLower(req.Email))
	if email == "" || req.Password == "" {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "email and password are required")
		return
	}

	user, err := h.Users.GetByEmail(c.Request.Context(), email)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to lookup user", err)
		return
	}
	if user == nil || user.PasswordHash == nil || !auth.CheckPassword(*user.PasswordHash, req.Password) {
		response.Fail(c, http.StatusUnauthorized, "UNAUTHORIZED", "email or password is incorrect")
		return
	}

	token, expiresAt, err := h.issueToken(user)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to issue token", err)
		return
	}

	response.OK(c, gin.H{
		"access_token": token,
		"token_type":   "Bearer",
		"expires_at":   expiresAt.UTC().Format(time.RFC3339),
		"user":         user.Public(),
	})
}

// Me godoc
//
//	@Summary		Current user profile
//	@Tags			auth
//	@Produce		json
//	@Param			Authorization	header		string	true	"Bearer access token"
//	@Success		200				{object}	response.Envelope
//	@Failure		401				{object}	response.Envelope
//	@Failure		404				{object}	response.Envelope
//	@Failure		500				{object}	response.Envelope
//	@Router			/api/v1/auth/me [get]
//	@Security		BearerAuth
func (h *Handler) Me(c *gin.Context) {
	id, ok := middleware.UserID(c)
	if !ok {
		response.Fail(c, http.StatusUnauthorized, "UNAUTHORIZED", "Bearer token required")
		return
	}
	user, err := h.Users.GetByID(c.Request.Context(), id)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to load user", err)
		return
	}
	if user == nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "user not found")
		return
	}
	response.OK(c, gin.H{"user": user.Public()})
}

// Logout godoc
//
//	@Summary		Logout (client discards token)
//	@Description	Stateless JWT — server acknowledges logout; FE must delete access_token.
//	@Tags			auth
//	@Produce		json
//	@Param			Authorization	header		string	true	"Bearer access token"
//	@Success		200				{object}	response.Envelope
//	@Failure		401				{object}	response.Envelope
//	@Router			/api/v1/auth/logout [post]
//	@Security		BearerAuth
func (h *Handler) Logout(c *gin.Context) {
	response.OK(c, gin.H{
		"logged_out": true,
		"note":       "discard access_token on client",
	})
}

func (h *Handler) issueToken(user *repository.User) (string, time.Time, error) {
	email := ""
	if user.Email != nil {
		email = *user.Email
	}
	return h.Tokens.Issue(user.ID, user.Role, email, user.FullName)
}
