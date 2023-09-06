# Download and install Miniforge (an equivalent of Miniconda)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O ~/miniforge.sh
bash ~/miniforge.sh -b -p ~/miniforge3

# Activate base env and run init for the future
source ~/miniforge3/etc/profile.d/conda.sh
conda activate
conda init

# Create conda environment
conda env create --file requirements.yaml

# Delete installer
rm ~/miniforge.sh