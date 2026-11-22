# Override with `make test PYTHON=python3.12` to use another environment.
PYTHON ?= .venv/bin/python

.PHONY: help check test lint format format-check build docs-check
.DEFAULT_GOAL := help

help:
	@echo "make check         Run lint, formatting checks, and the core test suite"
	@echo "make test          Run the core test suite (no model credentials needed)"
	@echo "make lint          Check Python imports and common errors"
	@echo "make format-check  Check Python formatting"
	@echo "make format        Format Python and sort imports"
	@echo "make docs-check    Check local documentation links"
	@echo "make build         Build and verify the wheel and source distribution"

check test lint format format-check build docs-check:
	$(PYTHON) tests/support/development.py $@
