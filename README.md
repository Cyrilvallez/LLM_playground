# LLM Playground

This repository is a minimal version of some other work, in order to easily interact with common open-source 
Large Language Models (LLMs) and to log the interactions. It is based on the `textwiz` library, a wrapper I created around Huggingface transformers library.

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

We use Gradio to easily create a web interface on which you can interact with the models. The syntax is the following:

```sh
python3 webapp_chat.py [--model model_name] [--int8] [--few_shot_template template.yaml] [--no_auth]
```

in order to launch the webapp. You should see both a local and public URL to access the webapp. 
You will be asked for your credentials when clicking on either link. You can also deactivate authentication 
using the `--no_auth` flag, but in this case, everyone with the public link will be able to access the webapp, even if you did not share the username and password with them.

If you pass the `--int8` flag, the model will use int8 quantization. The `--few_shot_template` flag is used in case you want to set a given system prompt and/or few shot examples.

This default can be used by multiple users at the same time, however the concurrency is set to 4, meaning that only 4 people can perform inference with the model at the same time (if more requests are sent, a queue is used).

## Usage: test of models

To easily experiment with multiple models, you can use:

```sh
python3 webapp.py [--no_auth]
```

In this webapp, you will have the opportunity to dynamically switch the model that is used for inference. 

⚠️⚠️⚠️ However, note that the ability to switch models on this demo comes at a cost. Indeed, only one model is shared between all users, so if multiple people use the demo at the same time, switching model while other are performing inference is likely to break the app.

## Logging

All interactions (input/output pairs) with the models will be saved into a csv file, namely `text_logs/log.csv`
for simple causal language modeling and into `chatbot_logs/log.csv` for conversations (chats) with the model.

## Computational power requirements

Finally, note that running LLMs is very costly in terms of computational power. For this reason, it is highly
recommended to run on a machine with AT LEAST 1 GPU. Depending on your hardware, you may not be able to load
some models, and/or inference on some models may be very slow.
