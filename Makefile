.DEFAULT_GOAL := help

IMAGE := support-query-classifier
TAG := dev
PORT := 8080

.PHONY: help install test lint format docker-build docker-run tf-init tf-plan tf-apply

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install dependencies (default groups)
	uv sync

test:  ## Run pytest with coverage
	uv run pytest

lint:  ## Run ruff check + mypy
	uv run ruff check .
	uv run mypy

format:  ## Auto-format with ruff
	uv run ruff format .

docker-build:  ## Build the serving Docker image
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE):$(TAG) .

docker-run:  ## Run the serving image locally on :8080
	docker run --rm -p $(PORT):$(PORT) $(IMAGE):$(TAG)

tf-init:  ## terraform init (in infra/)
	cd infra && terraform init

tf-plan:  ## terraform plan (in infra/)
	cd infra && terraform plan

tf-apply:  ## terraform apply (in infra/)
	cd infra && terraform apply
