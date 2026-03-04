.DEFAULT_GOAL := help

DOCKER_IMAGE := contentful-pinecone-embeddings

.PHONY: help install setup clear-embedding create-embedding docker-build docker-embed docker-clear-embedding

help:
	@echo ""
	@echo "Local:"
	@echo "  make install                         Install dependencies with uv"
	@echo "  make setup                           Create Pinecone index and clear namespace"
	@echo "  make clear-embedding <type>          Delete vectors for a content type from Pinecone"
	@echo "  make create-embedding <type>         Embed a content type locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build                    Build the Docker image"
	@echo "  make docker-clear-embedding <type>   Delete vectors for a content type (Docker)"
	@echo "  make docker-embed <type>             Run embedding pipeline in Docker"
	@echo ""
	@echo "  Available content types: article | event | course"
	@echo ""
	@echo "  Example (re-embed articles):"
	@echo "    make clear-embedding article && make create-embedding article"
	@echo "    make docker-clear-embedding article && make docker-embed article"
	@echo ""

install:
	uv sync

setup:
	PYTHONPATH=. uv run python setup_index.py

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
		-v hf-cache:/cache/huggingface \
		$(DOCKER_IMAGE) $(TYPE)

# Absorb extra goal arguments so Make doesn't error on unknown targets
%:
	@:
