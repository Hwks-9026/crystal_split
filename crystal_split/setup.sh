#!/bin/bash

[ ! -d ".venv" ] && python -m venv .venv

source .venv/bin/activate

[ ! -d ".venv/bin/maturin" ] && pip install -r requirements.txt

