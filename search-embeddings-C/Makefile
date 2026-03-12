.DEFAULT_GOAL := help

DOCKER_IMAGE := search-embedding-c

.PHONY: help install setup clear-embedding create-embedding \
        docker-build docker-setup docker-clear-embedding docker-embed

help:
	@echo ""
	@echo "Local:"
	@echo "  make install              Install dependencies with uv"
	@echo "  make setup               Create Pinecone index and clear namespace"
	@echo "  make clear-embedding <type>   Delete vectors for a content type from Pinecone"
	@echo "  make create-embedding <type>  Embed a content type locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build             Build the Docker image"
	@echo "  make docker-setup            Create index in Docker"
	@echo "  make docker-clear-embedding  Delete vectors for a content type (Docker)"
	@echo "  make docker-embed <type>     Embed a content type in Docker"
	@echo ""
	@echo "  Available content types: article | event | course"
	@echo ""
	@echo "  Typical workflow:"
	@echo "    make setup"
	@echo "    make create-embedding article"
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

docker-setup:
	docker run --rm --env-file .env $(DOCKER_IMAGE) setup

docker-clear-embedding:
	$(eval TYPE := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(TYPE)" ]; then \
		echo "❌ Specify a content type: make docker-clear-embedding <type>"; \
		echo "   Available: article | event | course"; \
		exit 1; \
	fi
	docker run --rm --env-file .env $(DOCKER_IMAGE) clear $(TYPE)

docker-embed:
	$(eval TYPE := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(TYPE)" ]; then \
		echo "❌ Specify a content type: make docker-embed <type>"; \
		echo "   Available: article | event | course"; \
		exit 1; \
	fi
	docker run --rm --env-file .env $(DOCKER_IMAGE) embed $(TYPE)

%:
	@:
