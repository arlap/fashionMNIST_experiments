VENV := .venv
BIN = .venv/bin/

init:
	/usr/local/opt/python@3.8/libexec/bin/python -m venv .venv
	$(BIN)pip3 install --upgrade pip
	$(BIN)pip3 install wheel
	$(BIN)pip3 install -r requirements.txt

data:

train:
