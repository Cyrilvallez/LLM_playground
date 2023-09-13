"""
This file contains the conversation templates for the models we use.
"""

from engine import loader

class GenericConversation(object):
    """Class used to store a conversation with a model."""

    def __init__(self, eos_token: str):

        # Conversation history
        self.user_history_text = []
        self.model_history_text = []

        # system prompt
        self.system_prompt = ''

        # eos token
        self.eos_token = eos_token

        # Extra eos tokens
        self.extra_eos_tokens = []


    def __len__(self) -> int:
        """Return the length of the current conversation.
        """
        return len(self.user_history_text)
    
    
    def __iter__(self):
        """Create a generator over (user_input, model_answer) tuples for all turns in the conversation.
        """
        # Generate over copies so that the object in the class cannot change during iteration
        for user_history, model_history in zip(self.user_history_text.copy(), self.model_history_text.copy()):
            yield user_history, model_history
    

    def __str__(self) -> str:
        """Format the conversation as a string.
        """

        N = len(self)

        if N == 0:
            return "The conversation is empty."
        
        else:
            out = ''
            for i, (user, model) in enumerate(self):
                out += f'>> User: {user}\n'
                if model is not None:
                    out += f'>> Model: {model}'
                # Skip 2 lines between turns
                if i < N - 1:
                    out += '\n\n'

            return out
        

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        

    def append_user_message(self, user_prompt: str):
        """Append a new user message, and set the corresponding answer of the model to `None`.

        Parameters
        ----------
        user_prompt : str
            The user message.
        """

        if None in self.model_history_text:
            raise ValueError('Cannot append a new user message before the model answered to the previous messages.')

        self.user_history_text.append(user_prompt)
        self.model_history_text.append(None)


    def append_model_message(self, model_output: str):
        """Append a new model message, by modifying the last `None` value in-place. Should always be called after
        `append_user_message`, before a new call to `append_user_message`.

        Parameters
        ----------
        model_output : str
            The model message.
        """

        if self.model_history_text[-1] is None:
            self.model_history_text[-1] = model_output
        else:
            raise ValueError('It looks like the last user message was already answered by the model.')
        

    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        # This seems to be the accepted way to treat inputs for conversation with a model that was not specifically
        # fine-tuned for conversation. This is the DialoGPT way of handling conversation, but is in fact reused by
        # all other tokenizers that we use.

        prompt = ''

        for user_message, model_response in self:

            prompt += user_message + self.eos_token
            if model_response is not None:
                prompt += model_response + self.eos_token

        return prompt
    

    def get_extra_eos(self) -> list[str]:
        return self.extra_eos_tokens
    

    def erase_conversation(self):
        """Reinitialize the conversation.
        """

        self.user_history_text = []
        self.model_history_text = []

    
    def set_conversation(self, past_user_inputs: list[str], past_model_outputs: list[str]):
        """Set the conversation.
        """

        self.user_history_text = past_user_inputs
        self.model_history_text = past_model_outputs


    def to_gradio_format(self) -> list[list[str, str]]:
        """Convert the current conversation to gradio chatbot format.
        """

        if len(self) == 0:
            return [[None, None]]

        return [list(conv_turn) for conv_turn in self]
        

# reference: https://huggingface.co/spaces/HuggingFaceH4/starchat-playground/blob/main/dialogues.py
class StarChatConversation(GenericConversation):

    def __init__(self, eos_token: str):

        super().__init__(eos_token)

        # Special tokens
        self.system_token = '<|system|>'
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'
        self.sep_token = '<|end|>'

        # extra eos
        self.extra_eos_tokens = [self.sep_token]


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """
        
        prompt = self.system_token + '\n' + self.system_prompt + self.sep_token + '\n'

        for user_message, model_response in self:

            prompt += self.user_token + '\n' + user_message + self.sep_token + '\n'
            if model_response is not None:
                prompt += self.assistant_token + '\n' + model_response + self.sep_token + '\n'
            else:
                prompt += self.assistant_token + '\n'

        return prompt
    

# reference: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L334
class VicunaConversation(GenericConversation):

    def __init__(self, eos_token: str):

        super().__init__(eos_token)

        self.system_prompt = ("A chat between a curious user and an artificial intelligence assistant. "
                              "The assistant gives helpful, detailed, and polite answers to the user's questions.")

        self.user_token = 'USER'
        self.assistant_token = 'ASSISTANT'


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        prompt = self.system_prompt + ' ' if self.system_prompt != '' else ''

        for user_message, model_response in self:

            prompt += self.user_token + ': ' + user_message + ' '
            if model_response is not None:
                prompt += self.assistant_token + ': ' + model_response + self.eos_token
            else:
                prompt += self.assistant_token + ':'

        return prompt
    

# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
class Llama2ChatConversation(GenericConversation):

    def __init__(self, eos_token: str):

        super().__init__(eos_token)

        self.bos_token = '<s>'

        self.system_template = '<<SYS>>\n{system_prompt}\n<</SYS>>\n\n'
        self.user_token = '[INST]'
        self.assistant_token = '[/INST]'


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        system_prompt = self.system_template.format(system_prompt=self.system_prompt.strip())
        prompt = ''

        for i, (user_message, model_response) in enumerate(self):

            if i == 0:
                # Do not add bos_token here as it will be added automatically at the start of the prompt by 
                # the tokenizer 
                prompt += self.user_token + ' ' + system_prompt + user_message.strip() + ' '
            else:
                prompt += self.bos_token + self.user_token + ' ' + user_message.strip() + ' '
            if model_response is not None:
                prompt += self.assistant_token + ' ' + model_response.strip() + ' ' + self.eos_token
            else:
                prompt += self.assistant_token

        return prompt
    


# Mapping from model name to conversation class name
CONVERSATION_MAPPING = {
    # StarChat
    'star-chat-alpha': StarChatConversation,
    'star-chat-beta': StarChatConversation,

    # Vicuna (1.3)
    'vicuna-7B': VicunaConversation,
    'vicuna-13B': VicunaConversation,

    # Llama2-chat
    'llama2-7B-chat': Llama2ChatConversation,
    'llama2-13B-chat': Llama2ChatConversation,
    'llama2-70B-chat': Llama2ChatConversation,

    # Code-llama-instruct
    'code-llama-7B-instruct': Llama2ChatConversation,
    'code-llama-13B-instruct': Llama2ChatConversation,
    'code-llama-34B-instruct': Llama2ChatConversation,
}


def get_conversation_template(model_name: str) -> GenericConversation:
    """Return the conversation object corresponding to `model_name`.

    Parameters
    ----------
    model_name : str
        Name of the current model.

    Returns
    -------
    GenericConversation
        A conversation object template class corresponding to `model_name`.

    """

    if model_name not in loader.ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*loader.ALLOWED_MODELS,}.')
    
    # TODO: maybe change this way of obtaining the eos token for a given model as it forces to load the
    # tokenizer for nothing (maybe create a mapping from name to eos?). For now it is sufficient as 
    # loading a tokenizer is sufficiently fast
    tokenizer = loader.load_tokenizer(model_name)
    eos_token = tokenizer.eos_token

    if model_name in CONVERSATION_MAPPING.keys():
        conversation = CONVERSATION_MAPPING[model_name](eos_token=eos_token)
    else:
        conversation = GenericConversation(eos_token=eos_token)

    return conversation