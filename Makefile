.PHONY: help install install-dev test test-cov lint format clean build docs

help:
	@echo "DistAwareAug - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        Install package in production mode"
	@echo "  make install-dev    Install package in development mode with all dependencies"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make lint           Run linting checks (flake8)"
	@echo "  make format         Format code with black and isort"
	@echo "  make clean          Remove build artifacts and cache files"
	@echo "  make build          Build distribution packages"
	@echo "  make docs           Build documentation (if available)"
	@echo "  make check          Run all checks (format, lint, test)"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -v

test-cov:
	pytest --cov=distawareaug --cov-report=html --cov-report=term-missing

lint:
	flake8 distawareaug tests --max-line-length=100

format:
	black distawareaug tests examples --line-length=100
	isort distawareaug tests examples --profile black

check: format lint test

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean
	python -m build

docs:
	@echo "Documentation generation not yet configured"
	@echo "TODO: Add Sphinx documentation"

publish: build
	@echo "Publishing to PyPI..."
	@echo "Make sure you have set up your PyPI credentials!"
	twine upload dist/*

publish-test: build
	@echo "Publishing to Test PyPI..."
	twine upload --repository testpypi dist/*
