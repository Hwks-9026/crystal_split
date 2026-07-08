#!/bin/bash

python -m venv .venv

source .venv/bin/activate

pip install --no-user -r requirements.txt
pip install maturin

cd diffraction_sim && maturin develop --release

echo "All Set Up!"
