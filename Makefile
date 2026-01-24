# Makefile for BrainStorm Track 2 project

.PHONY: help check-uv install sync setup-hooks download stream serve format lint type-check test test-cov check-all clean

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install        - Complete setup: check UV, sync dependencies, setup git hooks"
	@echo "  sync           - Sync/reinstall all dependencies"
	@echo "  setup-hooks    - Setup git pre-commit hooks"
	@echo "  download       - Download datasets from HuggingFace"
	@echo ""
	@echo "Running:"
	@echo "  stream         - Start the data stream (requires downloaded data)"
	@echo "  serve          - Start the web viewer server"
	@echo ""
	@echo "Development:"
	@echo "  format         - Format code with ruff"
	@echo "  lint           - Lint code with ruff"
	@echo "  type-check     - Run mypy type checking"
	@echo "  test           - Run tests with pytest"
	@echo "  test-cov       - Run tests with coverage"
	@echo "  check-all      - Run all checks (format, lint, type-check, test)"
	@echo "  clean          - Clean up __pycache__ and .pyc files"
	@echo ""

# Check if UV is installed
check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "❌ Error: UV is not installed!"; \
		echo ""; \
		echo "UV is required to manage dependencies for this project."; \
		echo "Please install UV using one of these methods:"; \
		echo ""; \
		echo "  macOS/Linux:"; \
		echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo ""; \
		echo "  Homebrew:"; \
		echo "    brew install uv"; \
		echo ""; \
		echo "  Windows:"; \
		echo "    powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""; \
		echo ""; \
		echo "For more information, visit: https://docs.astral.sh/uv/"; \
		echo ""; \
		exit 1; \
	}
	@echo "✓ UV is installed (version: $$(uv --version))"

# Sync dependencies (creates venv and installs everything)
sync: check-uv
	@echo "Syncing dependencies..."
	@uv sync
	@echo "✓ Dependencies synced"

# Setup git hooks
setup-hooks:
	@echo "Setting up pre-commit hooks..."
	@uv run pre-commit install
	@echo "✓ Pre-commit hooks installed"

# Complete installation
install: sync setup-hooks
	@echo ""
	@echo "✅ Installation complete!"
	@echo ""
	@echo "To run scripts, use 'uv run' or activate the venv:"
	@echo "  source .venv/bin/activate    # macOS/Linux"
	@echo ""
	@echo "Quick start:"
	@echo "  make download    # Download data"
	@echo "  make stream      # Start data stream"
	@echo "  make serve       # Start web viewer server"

# Download datasets from HuggingFace
download:
	@echo "Downloading datasets..."
	uv run python -m scripts.download easy
	@echo "✓ Downloaded easy dataset"
	@echo ""
	@echo "To download other difficulties:"
	@echo "  uv run python -m scripts.download medium"
	@echo "  uv run python -m scripts.download hard"

# Start data stream from downloaded files
stream:
	uv run brainstorm-stream --from-file data/easy/

# Start web viewer server
serve:
	uv run brainstorm-serve

# Code formatting
format:
	uv run ruff format .

# Linting
lint:
	uv run ruff check . --fix

# Type checking
type-check:
	uv run mypy scripts/

# Testing
test:
	uv run pytest

test-cov:
	uv run pytest --cov=scripts --cov-report=html --cov-report=term

# Run all checks
check-all: format lint type-check test

# Clean up Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

