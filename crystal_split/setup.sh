#!/bin/bash

[ ! -d ".venv" ] && python -m venv .venv

source .venv/bin/activate

[ ! -f ".venv/bin/maturin" ] && pip install -r requirements.txt

[ ! -d "./diffraction_sim/target/" ] && cd diffraction_sim && maturin develop --release

echo "All Set Up!"
