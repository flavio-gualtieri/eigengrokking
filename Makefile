PYTHON ?= python

# The models here are tiny: extra intra-op threads only add contention, and
# on CPU-quota'd login nodes they slow things down by an order of magnitude.
export OMP_NUM_THREADS ?= 1

.PHONY: install test smoke

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m eigengrok.train --config configs/smoke.yaml
