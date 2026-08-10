package auth

import (
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

const (
	RoleCitizen = "CITIZEN"
	RoleAdmin   = "ADMIN"
)

type Claims struct {
	Role  string `json:"role"`
	Email string `json:"email,omitempty"`
	Name  string `json:"name,omitempty"`
	jwt.RegisteredClaims
}

type TokenService struct {
	Secret []byte
	TTL    time.Duration
}

func HashPassword(password string) (string, error) {
	b, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func CheckPassword(hash, password string) bool {
	if hash == "" || password == "" {
		return false
	}
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}

func (s *TokenService) Issue(userID uuid.UUID, role, email, fullName string) (string, time.Time, error) {
	if len(s.Secret) < 16 {
		return "", time.Time{}, fmt.Errorf("jwt secret too short")
	}
	exp := time.Now().Add(s.TTL)
	claims := Claims{
		Role:  role,
		Email: email,
		Name:  fullName,
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   userID.String(),
			ExpiresAt: jwt.NewNumericDate(exp),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
		},
	}
	tok := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := tok.SignedString(s.Secret)
	if err != nil {
		return "", time.Time{}, err
	}
	return signed, exp, nil
}

func (s *TokenService) Parse(tokenString string) (*Claims, error) {
	tok, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(t *jwt.Token) (any, error) {
		if t.Method != jwt.SigningMethodHS256 {
			return nil, fmt.Errorf("unexpected signing method")
		}
		return s.Secret, nil
	})
	if err != nil {
		return nil, err
	}
	claims, ok := tok.Claims.(*Claims)
	if !ok || !tok.Valid {
		return nil, fmt.Errorf("invalid token")
	}
	return claims, nil
}
