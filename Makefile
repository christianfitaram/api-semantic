.PHONY: install lint test run bootstrap

install:
	poetry install --with dev

lint:
	poetry run ruff check .

test:
	poetry run pytest

run:
	poetry run uvicorn api_semantic.main:app --host 0.0.0.0 --port 8000

bootstrap:
	poetry run python -m api_semantic.bootstrap
