# LLM Playground

This repository is a minimal version of some other work, in order to easily interact with common open-source 
Large Language Models (LLMs) and to log the interactions.

## Environment setup

If you do not already have `conda` on your machine, please run the following command, which will install
`miniforge3` (for Linux) and create the correct `conda` environment:

```sh
cd path/to/this/repo
source config.sh
```

If you already have a `conda` installation, you can only run:

```sh
cd path/to/this/repo
conda env create --file requirements.yaml
```

## Authentication system

By default, we use a very simple authentication system in order to access the webapp. You will need to create
a file called `.gradio_login.txt` in the root of this repository, containing both a username and password
in two separate lines such as

```txt
username
password
````

Then you can use this username and password to connect to the webapp when prompted, and you can share them
with other people if you want them to try out the demo.

## Usage

We use Gradio to easily create a web interface on which you can interact with the models. By default,
simply run

```sh
python3 webapp.py
```

in order to launch the webapp. You should see both a local and public URL to access the webapp. 
You will be asked for your credentials when clicking on either link. You can also deactivate authentication 
using the `--no_auth` flag:

```sh
python3 webapp.py --no_auth
```

But in this case, everyone with the public link will be able to access the webapp, even if you did not share 
the username and password with them.

## Computational power requirements

Finally, note that running LLMs is very costly in terms of computational power. For this reason, it is highly
recommended to run on a machine with AT LEAST 1 GPU. Depending on your hardware, you may not be able to load
some models, and/or inference on some models may be very slow.
