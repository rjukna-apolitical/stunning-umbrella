.DEFAULT_GOAL := help

DOCKER_IMAGE := contentful-pinecone-embeddings-b

.PHONY: help install setup fit-bm25 clear-embedding create-embedding \
        docker-build docker-fit-bm25 docker-embed docker-clear-embedding

help:
	@echo ""
	@echo "Local:"
	@echo "  make install                         Install dependencies with uv"
	@echo "  make setup                           Create Pinecone index and clear namespace"
	@echo "  make fit-bm25                        Fit BM25 corpus stats (run once before embedding)"
	@echo "  make clear-embedding <type>          Delete vectors for a content type from Pinecone"
	@echo "  make create-embedding <type>         Embed a content type locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build                    Build the Docker image"
	@echo "  make docker-fit-bm25                 Fit BM25 corpus stats in Docker"
	@echo "  make docker-clear-embedding <type>   Delete vectors for a content type (Docker)"
	@echo "  make docker-embed <type>             Run embedding pipeline in Docker"
	@echo ""
	@echo "  Available content types: article | event | course"
	@echo ""
	@echo "  Typical workflow (first time):"
	@echo "    make setup"
	@echo "    make fit-bm25"
	@echo "    make create-embedding article"
	@echo ""
	@echo "  Re-embed a single type:"
	@echo "    make clear-embedding article && make create-embedding article"
	@echo ""

install:
	uv sync

setup:
	PYTHONPATH=. uv run python setup_index.py

fit-bm25:
	PYTHONPATH=. uv run python setup_bm25.py

clear-embedding:
	$(eval TYPE := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(TYPE)" ]; then \
		echo "❌ Specify a content type: make clear-embedding <type>"; \
		echo "   Available: article | event | course"; \
		exit 1; \
	fi
	PYTHONPATH=. uv run python -c "from modules.logger import setup_logging; setup_logging(); from modules.pinecone_utils import delete_by_type; delete_by_type('$(TYPE)')"

create-embedding:
	$(eval TYPE := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(TYPE)" ]; then \
		echo "❌ Specify a content type: make create-embedding <type>"; \
		echo "   Available: article | event | course"; \
		exit 1; \
	fi
	PYTHONPATH=. uv run python embed/$(TYPE).py

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-fit-bm25:
	docker run --rm \
		--env-file .env \
		-v bm25-data-b:/app/data \
		$(DOCKER_IMAGE) fit

docker-clear-embedding:
	$(eval TYPE := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(TYPE)" ]; then \
		echo "❌ Specify a content type: make docker-clear-embedding <type>"; \
		echo "   Available: article | event | course"; \
		exit 1; \
	fi
	docker run --rm \
		--env-file .env \
		$(DOCKER_IMAGE) clear $(TYPE)

docker-embed:
	$(eval TYPE := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(TYPE)" ]; then \
		echo "❌ Specify a content type: make docker-embed <type>"; \
		echo "   Available: article | event | course"; \
		exit 1; \
	fi
	docker run --rm \
		--env-file .env \
		-v bm25-data-b:/app/data \
		$(DOCKER_IMAGE) $(TYPE)

# Absorb extra goal arguments so Make doesn't error on unknown targets
%:
	@:
