.PHONY: install-uv install run-unit-test run-isort run-formatter clean

install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

install: install-uv
	uv sync

run-unit-test:
	uv run --extra dev pytest tests --verbose

run-isort:
	uv run --extra dev ruff check src tests --select I --fix

run-formatter:
	uv run --extra dev ruff format src tests

launch-jupyterlab:
	uv run jupyter lab --notebook-dir=workspace --ip=0.0.0.0 --no-browser

clean:
	rm -rf .pytest_cache .ruff_cache .venv build dist src/*.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
