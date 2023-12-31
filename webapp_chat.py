import gc
import os
import argparse
import queue
import copy
from concurrent.futures import ThreadPoolExecutor

from transformers import TextIteratorStreamer
import gradio as gr

from textwiz import HFModel, TextContinuationStreamer
from textwiz.conversation_template import GenericConversation, CONVERSATION_MAPPING
import textwiz.web_interface as wi
from helpers import utils

# Default model to load at start-up
DEFAULT = 'llama2-7B-chat'

# All the chat models we allow
ALLOWED_MODELS = list(CONVERSATION_MAPPING.keys())

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# This will be a mapping between users and current conversation, to reload them with page reload
CACHED_CONVERSATIONS = {}

# Need to define one logger per user
LOGGERS = {}


def chat_generation(conversation: GenericConversation, prompt: str, max_new_tokens: int, do_sample: bool,
                    top_k: int, top_p: float, temperature: float, use_seed: bool,
                    seed: int) -> tuple[str, GenericConversation, list[list]]:
    yield from wi.chat_generation(MODEL, conversation=conversation, prompt=prompt, max_new_tokens=max_new_tokens,
                                  do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature, use_seed=use_seed,
                                  seed=seed)


def continue_generation(conversation: GenericConversation, additional_max_new_tokens: int, do_sample: bool,
                        top_k: int, top_p: float, temperature: float, use_seed: bool,
                        seed: int) -> tuple[GenericConversation, list[list]]:
    yield from wi.continue_generation(MODEL, conversation=conversation, additional_max_new_tokens=additional_max_new_tokens,
                                      do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                      use_seed=use_seed, seed=seed)


def retry_chat_generation(conversation: GenericConversation, max_new_tokens: int, do_sample: bool,
                          top_k: int, top_p: float, temperature: float, use_seed: bool,
                          seed: int) -> tuple[GenericConversation, list[list]]:
    yield from wi.retry_chat_generation(MODEL, conversation=conversation, max_new_tokens=max_new_tokens,
                                        do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                        use_seed=use_seed, seed=seed)


def authentication(username: str, password: str) -> bool:
    return wi.simple_authentication(CREDENTIALS_FILE, username, password)
    

def clear_chatbot(username: str) -> tuple[GenericConversation, str, list[list]]:
    """Erase the conversation history and reinitialize the elements.

    Parameters
    ----------
    username : str
        The username of the current session if any.

    Returns
    -------
    tuple[GenericConversation, str, list[list]]
        Corresponds to the tuple of components (conversation, output, conv_id)
    """

    # Create new conv object (we need a new unique id)
    conversation = MODEL.get_conversation_from_yaml_template(TEMPLATE_PATH) if USE_TEMPLATE else MODEL.get_empty_conversation()
    if username != '':
        CACHED_CONVERSATIONS[username] = conversation

    return conversation, conversation.to_gradio_format(), conversation.id



def loading(request: gr.Request) -> tuple[GenericConversation, list[list], str, str, dict]:
    """Retrieve username and all cached values at load time, and set the elements to the correct values.

    Parameters
    ----------
    request : gr.Request
        Request sent to the app.

    Returns
    -------
    tuple[GenericConversation, list[list], str, str, dict]
        Corresponds to the tuple of components (conversation, output, conv_id, username, max_new_tokens)
    """

    # Retrieve username
    if request is not None:
        try:
            username = request.username
        except:
            username = ''
    
    if username is None:
        username = ''
    
    # Check if we have cached a value for the conversation to use
    if username != '':
        if username in CACHED_CONVERSATIONS.keys():
            actual_conv = CACHED_CONVERSATIONS[username]
        else:
            actual_conv = MODEL.get_conversation_from_yaml_template(TEMPLATE_PATH) if USE_TEMPLATE else MODEL.get_empty_conversation()
            CACHED_CONVERSATIONS[username] = actual_conv
            LOGGERS[username] = gr.CSVLogger()
        
        LOGGERS[username].setup(inputs_to_callback, flagging_dir=f'chatbot_logs/{username}')

    # In this case we do not know the username so we don't store the conversation in cache
    else:
        actual_conv = MODEL.get_conversation_from_yaml_template(TEMPLATE_PATH) if USE_TEMPLATE else MODEL.get_empty_conversation()
        if username not in LOGGERS.keys():
            LOGGERS[username] = gr.CSVLogger()
        LOGGERS[username].setup(inputs_to_callback, flagging_dir='chatbot_logs/UNKNOWN')

    conv_id = actual_conv.id
    
    return actual_conv, actual_conv.to_gradio_format(), conv_id, username, gr.update(maximum=MODEL.get_context_size())
    

# Define general elements of the UI (generation parameters)
max_new_tokens = gr.Slider(32, 4096, value=512, step=32, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
max_additional_new_tokens = gr.Slider(16, 1028, value=128, step=16, label='Max additional new tokens',
                           info='New tokens to generate with "Continue last answer".')
do_sample = gr.Checkbox(value=True, label='Sampling', info=('Whether to incorporate randomness in generation. '
                                                            'If not selected, perform greedy search.'))
top_k = gr.Slider(0, 200, value=50, step=5, label='Top-k',
               info='How many tokens with max probability to consider. 0 to deactivate.')
top_p = gr.Slider(0, 1, value=0.90, step=0.01, label='Top-p',
              info='Probability density threshold for new tokens. 1 to deactivate.')
temperature = gr.Slider(0, 1, value=0.8, step=0.01, label='Temperature',
                        info='How to cool down the probability distribution.')
use_seed = gr.Checkbox(value=False, label='Use seed', info='Whether to use a fixed seed for reproducibility.')
seed = gr.Number(0, label='Seed', info='Seed for reproducibility.', precision=0)

# Define elements of the chatbot Tab
prompt = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
output = gr.Chatbot(label='Conversation', height=500)
generate_button = gr.Button('▶️ Submit', variant='primary')
continue_button = gr.Button('🔂 Continue', variant='primary')
retry_button = gr.Button('🔄 Retry', variant='primary')
clear_button = gr.Button('🗑 Clear')

# Initial value does not matter -> will be set correctly at loading time
conversation = gr.State(GenericConversation('</s>'))
# Define NON-VISIBLE elements: they are only used to keep track of variables and save them to the callback (States
# cannot be used in callbacks).
username = gr.Textbox('', label='Username', visible=False)
conv_id = gr.Textbox('', label='Conversation id', visible=False)

# Define the inputs for the main inference
inputs_to_chatbot = [conversation, prompt, max_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]
inputs_to_chatbot_continuation = [conversation, max_additional_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]
inputs_to_chatbot_retry = [conversation, max_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]

# Define inputs for the logging callbacks
inputs_to_callback = [username, output, conv_id, max_new_tokens, max_additional_new_tokens, do_sample,
                      top_k, top_p, temperature, use_seed, seed]

# Some prompt examples
prompt_examples = [
    "Please write a function to multiply 2 numbers `a` and `b` in Python.",
    "Hello, what's your name?",
    "What's the meaning of life?",
    "How can I write a Python function to generate the nth Fibonacci number?",
    ("Here is my data {'Name':['Tom', 'Brad', 'Kyle', 'Jerry'], 'Age':[20, 21, 19, 18], 'Height' :"
     " [6.1, 5.9, 6.0, 6.1]}. Can you provide Python code to plot a bar graph showing the height of each person?"),
]


demo = gr.Blocks(title='Text generation with LLMs')

with demo:

    # state variables
    conversation.render()
    username.render()
    conv_id.render()

    # Main UI
    output.render()
    prompt.render()

    with gr.Row():
        generate_button.render()
        continue_button.render()
        retry_button.render()
        clear_button.render()
            
    # Accordion for generation parameters
    with gr.Accordion("Text generation parameters", open=False):
        do_sample.render()
        with gr.Group():
            max_new_tokens.render()
            max_additional_new_tokens.render()
        with gr.Group():
            top_k.render()
            top_p.render()
            temperature.render()
        with gr.Group():
            use_seed.render()
            seed.render()

    gr.Markdown("### Prompt Examples")
    gr.Examples(prompt_examples, inputs=prompt)



    # Perform chat generation when clicking the button
    generate_event1 = generate_button.click(chat_generation, inputs=inputs_to_chatbot,
                                                 outputs=[prompt, conversation, output])

    # Add automatic callback on success
    generate_event1.success(lambda *args: LOGGERS[args[0]].flag(args, flag_option='generation'),
                            inputs=inputs_to_callback, preprocess=False)
    
    # Continue generation when clicking the button
    generate_event2 = continue_button.click(continue_generation, inputs=inputs_to_chatbot_continuation,
                                                 outputs=[conversation, output])
    
    # Add automatic callback on success
    generate_event2.success(lambda *args: LOGGERS[args[0]].flag(args, flag_option='continuation'),
                            inputs=inputs_to_callback, preprocess=False)
    
    # Continue generation when clicking the button
    generate_event3 = retry_button.click(retry_chat_generation, inputs=inputs_to_chatbot_retry,
                                              outputs=[conversation, output])
    
    # Add automatic callback on success
    generate_event3.success(lambda *args: LOGGERS[args[0]].flag(args, flag_option='retry'),
                            inputs=inputs_to_callback, preprocess=False)
    
    # Clear the prompt and output boxes when clicking the button
    clear_button.click(clear_chatbot, inputs=[username], outputs=[conversation, output, conv_id], queue=False)

    # Change visibility of generation parameters if we perform greedy search
    do_sample.input(lambda value: [gr.update(visible=value) for _ in range(5)], inputs=do_sample,
                    outputs=[top_k, top_p, temperature, use_seed, seed], queue=False)
    
    # Correctly display the model and quantization currently on memory if we refresh the page (instead of default
    # value for the elements) and correctly reset the chat output
    loading_events = demo.load(loading, outputs=[conversation, output, conv_id, username, max_new_tokens], queue=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Playground')
    parser.add_argument('--model', type=str, default=DEFAULT, choices=ALLOWED_MODELS,
                        help='The model to use.')
    parser.add_argument('--gpu_rank', type=int, default=1,
                        help='The gpu to use (if only one gpu is needed).')
    parser.add_argument('--int8', action='store_true',
                        help='Whether to quantize the model to Int8.')
    parser.add_argument('--few_shot_template', type=str, default='None',
                        help='Name of a yaml file containing the few shot examples to use.')
    parser.add_argument('--no_auth', action='store_true',
                        help='If given, will NOT require authentication to access the webapp.')
    
    args = parser.parse_args()
    no_auth = args.no_auth
    model = args.model
    rank = args.gpu_rank
    int8 = args.int8

    # We assume that if we use int8 quantization, we will need only 1 GPU and directly set it to the only device
    if int8:
        utils.set_cuda_visible_device(rank)
        # Because it is implicitly re-ranked as 0
        rank = 0

    # Check if we are going to use a few shot example
    TEMPLATE_NAME = args.few_shot_template
    if '/' in TEMPLATE_NAME:
        raise ValueError('The template name should not contain any "/".')
    TEMPLATE_PATH = os.path.join(utils.FEW_SHOT_FOLDER, TEMPLATE_NAME)
    USE_TEMPLATE = False if TEMPLATE_NAME == 'None' else True

    # Initialize global model (necessary not to reload the model for each new inference)
    MODEL = HFModel(model, gpu_rank=rank, quantization_8bits=int8)
    
    if no_auth:
        demo.queue(concurrency_count=4).launch(share=True, blocked_paths=[CREDENTIALS_FILE])
    else:
        demo.queue(concurrency_count=4).launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])
