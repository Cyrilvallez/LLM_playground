conda env create -f requirements.yaml
# This needs to be run after as we cannot specify the --no-build-isolation otherwise
pip install flash-attn --no-build-isolation