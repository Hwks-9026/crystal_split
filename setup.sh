#!/bin/bash
set -e

#module load python

python -m venv .venv


# NECCESARY C LIBRARIES
./get_libffi.sh
./get_libssl.sh
# ADD PATH EXPORT TO VENV, EACH TIME VENV IS ACTIVATED, LIBRARIES WILL BE LOADED
grep -qxF 'export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' .venv/bin/activate || \
  echo 'export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> .venv/bin/activate

source .venv/bin/activate
pip install --upgrade pip

pip install --no-user -r requirements.txt
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ IF RUNNING INTO 'OUTDATED CUDA VERSION', ADD --force-reinstall --no-cache-dir

pip install maturin

cd diffraction_sim && maturin develop --release


echo "All Set Up!"
