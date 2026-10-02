# Web — Vite static build served by nginx
# Build from repo root:
#   docker build -f apps/web/Dockerfile -t cas-web:latest .

FROM node:22-alpine AS build
WORKDIR /src
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
ARG VITE_ADMIN_INGESTION=false
ARG VITE_ADMIN_INDEXING=false
ENV VITE_ADMIN_INGESTION=$VITE_ADMIN_INGESTION
ENV VITE_ADMIN_INDEXING=$VITE_ADMIN_INDEXING
RUN npm run build

FROM nginx:1.27-alpine
COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 80
