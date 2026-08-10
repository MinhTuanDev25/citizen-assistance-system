package middleware

import (
	"net/http"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const (
	ContextUserID    = "auth_user_id"
	ContextUserRole  = "auth_user_role"
	ContextUserEmail = "auth_user_email"
	ContextUserName  = "auth_user_name"
)

// OptionalJWT parses Bearer token when present; ignores missing/invalid.
func OptionalJWT(tokens *auth.TokenService) gin.HandlerFunc {
	return func(c *gin.Context) {
		if claims := parseBearer(c, tokens); claims != nil {
			setClaims(c, claims)
		}
		c.Next()
	}
}

// RequireJWT requires a valid Bearer access token.
func RequireJWT(tokens *auth.TokenService) gin.HandlerFunc {
	return func(c *gin.Context) {
		h := strings.TrimSpace(c.GetHeader("Authorization"))
		if h == "" || !strings.HasPrefix(strings.ToLower(h), "bearer ") {
			abortUnauthorized(c, "Bearer token required")
			return
		}
		claims := parseBearer(c, tokens)
		if claims == nil {
			abortUnauthorized(c, "invalid or expired token")
			return
		}
		setClaims(c, claims)
		c.Next()
	}
}

// RequireAdmin requires a valid JWT with role ADMIN.
func RequireAdmin(tokens *auth.TokenService) gin.HandlerFunc {
	return func(c *gin.Context) {
		h := strings.TrimSpace(c.GetHeader("Authorization"))
		if h == "" || !strings.HasPrefix(strings.ToLower(h), "bearer ") {
			abortUnauthorized(c, "Bearer token required")
			return
		}
		claims := parseBearer(c, tokens)
		if claims == nil {
			abortUnauthorized(c, "invalid or expired token")
			return
		}
		if claims.Role != auth.RoleAdmin {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"request_id": GetRequestID(c),
				"error": gin.H{
					"code":    "FORBIDDEN",
					"message": "admin role required",
				},
			})
			return
		}
		setClaims(c, claims)
		c.Next()
	}
}

func abortUnauthorized(c *gin.Context, message string) {
	c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
		"request_id": GetRequestID(c),
		"error": gin.H{
			"code":    "UNAUTHORIZED",
			"message": message,
		},
	})
}

func parseBearer(c *gin.Context, tokens *auth.TokenService) *auth.Claims {
	if tokens == nil {
		return nil
	}
	h := strings.TrimSpace(c.GetHeader("Authorization"))
	if h == "" || !strings.HasPrefix(strings.ToLower(h), "bearer ") {
		return nil
	}
	raw := strings.TrimSpace(h[7:])
	if raw == "" {
		return nil
	}
	claims, err := tokens.Parse(raw)
	if err != nil {
		return nil
	}
	if _, err := uuid.Parse(claims.Subject); err != nil {
		return nil
	}
	return claims
}

func setClaims(c *gin.Context, claims *auth.Claims) {
	c.Set(ContextUserID, claims.Subject)
	c.Set(ContextUserRole, claims.Role)
	c.Set(ContextUserEmail, claims.Email)
	c.Set(ContextUserName, claims.Name)
}

func UserID(c *gin.Context) (uuid.UUID, bool) {
	v, ok := c.Get(ContextUserID)
	if !ok {
		return uuid.Nil, false
	}
	s, ok := v.(string)
	if !ok {
		return uuid.Nil, false
	}
	id, err := uuid.Parse(s)
	if err != nil {
		return uuid.Nil, false
	}
	return id, true
}

func UserRole(c *gin.Context) string {
	if v, ok := c.Get(ContextUserRole); ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}
