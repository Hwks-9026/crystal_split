# Crystal Split

Crystal Split is a machine learning pipeline designed to isolate diffraction patterns from images with multiple crystal lattices.
<img width="960" height="720" alt="Multi-Lattice Separation Example" src="https://github.com/user-attachments/assets/e4e80853-a884-4cd7-a479-d4ca791ebfd1" />

A ResNet50 based U-Net is used to predict a mask for each lattice pattern based on on-the-fly training data created by a custom library written in rust: `./diffraction_sim/`.
<img width="960" height="720" alt="Peak Finding Example" src="https://github.com/user-attachments/assets/123d96b3-e208-4344-924c-acbc418327de" />

___

## Features
- Iterative Lattice Extraction: The model processes each image multiple times to extract one lattice at a time.
- Hybrid Peak Detection: Uses U-Net probability output followed by generic peak finding to eliminate unwanted features like ice rings, beam guards, etc.
- High Performance Data Generation: Uses custom rust back end to enable rapid multi-threaded artificial data creation.
- 
___

## Requirements
- python 3.11 or later
- The Rust toolchain

___

## How To Use

### 1
Setup the environment by running [./setup.sh].
There are a few commented-out options in the script depending on the system you're running. 
> [!NOTE]
> If you're using a version of CUDA other than 12.6, change the source for PyTorch in `./requirements.txt`

The setup script will:
- Create and enter a virtual environment
- Install all requirements
- Compile the Diffraction Sim library for use with python. 

If you return to this program multiple times, be sure to run:
```bash
$ bash
$ source .venv/bin/activate
$ [PREFERRED SHELL]
```

### 2
Train the model with [./train.py]
The training can be interrupted at any time with ^C, and the script will save the most recent version. 
The script will also terminate after 100 Epochs.
While [./train.py] is running, it saves snapshots of its progress as images, which can be viewed with [./view_training.py] to observe the model's progress.

A .pth file will be saved with the model when the script terminates.

### 3
Test the model with [./test.py]
Currently, the model is also tested on simulated data, which can be seen detail through [./test.py]. 
The model is tested on its ability to split multi-lattice images and isolate all peaks in the same image, one example at a time.



