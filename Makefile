PRECOMMIT_VERSION="2.12.0"

.PHONY: pre-commit
pre-commit:
	wget -O pre-commit.pyz https://github.com/pre-commit/pre-commit/releases/download/v${PRECOMMIT_VERSION}/pre-commit-${PRECOMMIT_VERSION}.pyz
	python3 pre-commit.pyz install
	python3 pre-commit.pyz install --hook-type commit-msg
	rm pre-commit.pyz
