.PHONY: dev requirements prod setup

setup: env

requirements:
	source env/bin/activate && pip freeze > requirements.txt

env:
	python3 -m venv env
	source env/bin/activate && pip install -r requirements.txt

dev: env
	source env/bin/activate && FLASK_APP=server.py FLASK_ENV=development flask run
prod: env
	python server.py
