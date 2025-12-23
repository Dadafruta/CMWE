SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# Lets us avoid tabs in recipes (copy/paste friendly)
.RECIPEPREFIX := >

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install preflight smoke clean

$(PYTHON):
> python3 -m venv $(VENV)

install: $(PYTHON)
> $(PIP) install -U pip wheel setuptools
> $(PIP) install -r requirements.txt

preflight: install
> $(PYTHON) scripts/preflight.py

smoke: install
> PYTHON="$(PYTHON)" bash scripts/smoke.sh

clean:
> rm -rf $(VENV) .mypy_cache .ruff_cache __pycache__ */__pycache__
