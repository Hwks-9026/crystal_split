#!/bin/bash
set -e

DIALS_DIR="$(pwd)/dials-env"

if [ ! -f "$DIALS_DIR/dials_env.sh" ]; then
    echo "=> Downloading and installing DIALS bundle..."
    wget -nc https://dials.diamond.ac.uk/diamond_builds/dials-linux-x86_64-conda3.tar.xz
    
    mkdir -p dials-installer
    tar -xJf dials-linux-x86_64-conda3.tar.xz -C dials-installer --strip-components=1
    
    cd dials-installer
    ./install --prefix="$DIALS_DIR"
    cd ..
    
    rm -rf dials-installer dials-linux-x86_64-conda3.tar.xz
else
    echo "=> DIALS environment already installed at $DIALS_DIR."
fi


./get_libffi.sh
./get_libssl.sh


grep -qxF 'export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' "$DIALS_DIR/dials_env.sh" || \
  echo 'export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> "$DIALS_DIR/dials_env.sh"


source "$DIALS_DIR/dials_env.sh"


echo "=> Installing Python requirements via pip..."
pip install --upgrade pip

pip install --no-user -r requirements.txt
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ IF RUNNING INTO 'OUTDATED CUDA VERSION', ADD --force-reinstall --no-cache-dir

pip install maturin


echo "=> Building diffraction_sim..."
cd diffraction_sim && maturin develop --release
cd ..

echo "All Set Up! Remember to run 'source dials-env/dials_env.sh' to activate this environment in the future."
