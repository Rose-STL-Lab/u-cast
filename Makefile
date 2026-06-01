.PHONY: quality style test

check_dirs := src tests

quality:
	python3 -m black --check $(check_dirs) run.py run_inference_standalone.py
	python3 -m ruff check $(check_dirs) run.py run_inference_standalone.py

style:
	python3 -m black $(check_dirs) run.py run_inference_standalone.py
	python3 -m ruff check $(check_dirs) run.py run_inference_standalone.py --fix

test:
	python3 -m pytest tests/
