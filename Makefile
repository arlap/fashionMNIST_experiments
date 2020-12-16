VENV := .venv
BIN = .venv/bin/

init:
	python3 -m venv .venv
	$(BIN)pip3 install --upgrade pip
	$(BIN)pip3 install wheel
	$(BIN)pip3 install -r requirements.txt


