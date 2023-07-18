TIMEOUT ?= 300

.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	python docs/notebook_version_standardizer.py check-versions
	python docs/notebook_version_standardizer.py check-execution
	black . --check --config=./pyproject.toml
	ruff . --config=./pyproject.toml

.PHONY: lint-fix
lint-fix:
	python docs/notebook_version_standardizer.py standardize
	black . --config=./pyproject.toml
	ruff . --config=./pyproject.toml --fix

.PHONY: installdeps
installdeps:
	pip install --upgrade pip
	pip install -e .

.PHONY: installdeps-min
installdeps-min:
	pip install --upgrade pip -q
	pip install -e . --no-dependencies
	pip install -r tests/dependency_update_check/minimum_test_requirements.txt
	pip install -r tests/dependency_update_check/minimum_requirements.txt

.PHONY: installdeps-prophet
installdeps-prophet:
	pip install -e .[prophet]

.PHONY: installdeps-test
installdeps-test:
	pip install -e .[test]

.PHONY: installdeps-dev
installdeps-dev:
	pip install -e .[dev]
	pre-commit install

.PHONY: installdeps-docs
installdeps-docs:
	pip install -e .[docs]

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	python -m pip install --upgrade build

.PHONY: upgradesetuptools
upgradesetuptools:
	python -m pip install --upgrade setuptools
