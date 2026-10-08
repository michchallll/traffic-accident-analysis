PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

.PHONY: install test part01 part02 final clean

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) part01/test_part01.py

part01:
	$(PYTHON) part01/part01.py

part02:
	$(PYTHON) part02/analysis.py

final:
	$(PYTHON) final/geo.py
	$(PYTHON) final/doc.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
