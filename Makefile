# Makefile
.PHONY: help install install-dev test lint format clean build upload docs

help:
	@echo "Available commands:"
	@echo "  make install      Install the package"
	@echo "  make install-dev  Install package in development mode with dev dependencies"
	@echo "  make test         Run unit tests"
	@echo "  make lint         Run code linting"
	@echo "  make format       Format code with black"
	@echo "  make clean        Clean build artifacts"
	@echo "  make build        Build distribution packages"
	@echo "  make upload       Upload to PyPI (requires credentials)"
	@echo "  make docs         Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pytest pytest-cov black flake8 mypy

test:
	pytest tests/ -v --cov=malaria_forecast --cov-report=html --cov-report=term

lint:
	flake8 malaria_forecast/ tests/
	mypy malaria_forecast/ --ignore-missing-imports

format:
	black malaria_forecast/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

docs:
	cd docs && make html
