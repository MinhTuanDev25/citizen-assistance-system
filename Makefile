.PHONY: db-up db-down db-migrate db-migrate-down db-psql db-reset api-run api-tidy api-swagger web-run

db-up:
	cd deploy && docker compose up -d postgres

db-down:
	cd deploy && docker compose down

db-migrate:
	cd deploy && ./scripts/migrate.sh up

db-migrate-down:
	cd deploy && ./scripts/migrate.sh down 1

db-psql:
	cd deploy && docker compose exec postgres psql -U cas -d citizen_assistance

db-reset:
	cd deploy && docker compose down -v && docker compose up -d postgres && ./scripts/migrate.sh up

api-tidy:
	cd apps/api && go mod tidy

api-swagger:
	cd apps/api && swag init -g cmd/api/main.go -o docs --parseDependency --parseInternal

api-run:
	cd apps/api && APP_ENV=local go run ./cmd/api

web-run:
	cd apps/web && npm install && npm run dev
