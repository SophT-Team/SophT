#!/usr/bin/env just --justfile

_clean_pattern pattern:
    find . -name "{{ pattern }}" -prune -exec rm -rf {} +

clean_ds_store:
    @just --justfile {{ justfile() }} _clean_pattern ".DS_Store"

clean_mypy_cache:
    @just --justfile {{ justfile() }} _clean_pattern ".mypy_cache"

clean_pycache:
    @just --justfile {{ justfile() }} _clean_pattern "__pycache__"

clean_pytest_cache:
    @just --justfile {{ justfile() }} _clean_pattern ".pytest_cache"

clean_ruff_cache:
    @just --justfile {{ justfile() }} _clean_pattern ".ruff_cache"

clean: clean_ds_store clean_mypy_cache clean_pycache clean_pytest_cache clean_ruff_cache

format:
    ruff format sopht tests examples

lint:
    ruff check --fix sopht tests examples

sys_info:
    @echo "Operating System: {{ os() }}"
    @echo "Architecture: {{ arch() }}"
    @python_cmd="$(command -v python || command -v python3)"; \
        echo "Python binary: ${python_cmd}"; \
        echo "Python version: $($python_cmd -V 2>&1)"

test:
    @pytest tests --disable-warnings -v

typecheck:
    @mypy sopht
