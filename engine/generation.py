import os
import warnings
import gc
import psutil
import math
import copy

import torch
import numpy as np
from transformers import StoppingCriteriaList

from engine import loader
from engine import stopping
from engine.prompt_template import GenericPromptTemplate, get_prompt_template
from engine.conversation_template import GenericConversation, get_conversation_template
from engine.code_parser import CodeParser
from helpers import utils


class HFModel(object):
    """Class encapsulating a HuggingFace model and its tokenizer to generate text. 
    """

    def __init__(self, model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
                 device_map: dict | None = None, gpu_rank: int = 0):
        
        # Save the current allocated memory on each gpu to estimate model size after loading
        if torch.cuda.is_available():
            reference_memory = {}
            for i in range(torch.cuda.device_count()):
                reference_memory[i] = torch.cuda.memory_allocated(i)

        # Actually load the model and tokenizer
        self.model, self.tokenizer = loader.load_model_and_tokenizer(model_name, quantization_8bits=quantization_8bits,
                                                                     quantization_4bits=quantization_4bits, dtype=dtype,
                                                                     max_fraction_gpu_0=max_fraction_gpu_0,
                                                                     max_fraction_gpus=max_fraction_gpus,
                                                                     device_map=device_map, gpu_rank=gpu_rank)
        
        # Compute the memory footprint of the model on each gpu
        self.gpu_memory_map = {}

        # In this case, the model is on multiple devices
        if hasattr(self.model, 'hf_device_map'):
            self.device_map = self.model.hf_device_map

            gpu_devices = set(self.model.hf_device_map.values())
            gpu_devices.discard('cpu')
            gpu_devices.discard('disk')

            # Compute the gpus memory footprint
            self.input_device = min(gpu_devices) if len(gpu_devices) > 0 else 'cpu'
            for device in gpu_devices:
                self.gpu_memory_map[device] = (torch.cuda.memory_allocated(device) - reference_memory[device]) / 1024**3
        
        # In this case, the model is on a single device
        else:
            device = next(self.model.parameters()).get_device()
            self.device_map = 'cpu' if device == -1 else f'cuda:{device}'
            self.input_device = 'cpu' if device == -1 else device

            # Compute the gpu memory if the device is a gpu
            if device != -1:
                self.gpu_memory_map[device] = (torch.cuda.memory_allocated(device) - reference_memory[device]) / 1024**3

        # Maximum memory taken by the model on gpus, or on the cpu
        if len(self.gpu_memory_map) > 0:
            self.max_memory_footprint = max(self.gpu_memory_map.values())
        else:
            # Estimate the footprint via the number of parameters in this case
            self.max_memory_footprint = self.model.get_memory_footprint() / 1024**3

        self.model_name = model_name
        self.quantization_8bits = quantization_8bits
        self.quantization_4bits = quantization_4bits
        # May be different from the dtype given in the arguments so use the model attribute
        self.dtype = self.model.dtype


        # Initialize the prompt template to use 
        self.prompt_template = get_prompt_template(self.model_name)

        # Extra eos tokens
        self.extra_eos_tokens = self.prompt_template.get_extra_eos()

        # Flag to check if the model is a chat model by default
        self._is_chat_model = self.prompt_template.default_mode == 'chat'

    
    def dtype_category(self) -> str:
        """Return a string representation of the model dtype."""
        if self.quantization_4bits:
            return 'int4'
        elif self.quantization_8bits:
            return 'int8'
        else:
            return str(self.dtype).split('.', 1)[1]

    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.model_name}, quantization_8bits={self.quantization_8bits}, '
                f'quantization_4bits={self.quantization_4bits}, dtype={self.dtype})')
    
    
    def __str__(self) -> str:
        return f'{self.model_name}, with dtype {self.dtype_category()}'
        
        
    def is_chat_model(self) -> bool:
        """Check if the model was originally optimized as a chat agent."""
        return self._is_chat_model
    

    def get_gpu_memory_footprint(self) -> dict:
        """Return the memory footprint of the model on each GPU device it uses, in GiB."""
        return copy.deepcopy(self.gpu_memory_map)
    

    def get_memory_footprint(self) -> dict:
        """Return the memory footprint of the model on each device it uses, in GiB. In case of a custom `device_map`
        where both gpu devices AND cpu and/or disk were specified, this function is not accurate.
        """

        gpu_footprint = self.get_gpu_memory_footprint()
        if len(gpu_footprint) == 0:
            return {'cpu': self.max_memory_footprint}
        else:
            # If the custom device map contains both gpu device and cpu (and/or disk), this is not accurate as
            # we only return the footprint of the gpus (computing the footprint of the cpu is hard and not
            # precise, and this case should never appear in practice)
            return gpu_footprint
        

    def get_max_device_memory_footprint(self) -> float:
        """Return the maximum (accross devices) memory used by the model."""
        return self.max_memory_footprint
    

    def get_gpu_devices(self) -> tuple[int]:
        """Return the gpu devices used by the model."""
        return tuple(sorted(self.gpu_memory_map.keys()))


    def format_prompt(self, prompt: str, model_context: str = '', infill_suffix: str = '', system_prompt: str = '',
                      prompt_template_mode: str = 'default') -> str:
        """Format the prompt according to the prompt template.

        Parameters
        ----------
        prompt : str
            The prompt to the model.
        model_context : str, optional
            An optional context forming the start of the model answer. For `generation` mode, this is simply
            appended to `prompt`. By default ''.
        infill_suffix : str, optional
            An optional suffix to form the prompt. This is ignored for all `prompt_template_mode` except
            `infill`, by default ''.
        system_prompt : str, optional
            An optional system prompt to append at the beginning for chat mode. This is ignored for all 
            `prompt_template_mode` except `chat`, by default ''.
        prompt_template_mode: str
            The template mode for formatting the prompt. One of `('default', 'generation', 'infill', 'chat')`.
            Note that changing this value may result in errors or inconsistent results as usually a model is
            optimized for only one given prompt format. By default 'default', which chooses the best mode for
            the current model.

        Returns
        -------
        str
            The formatted prompt to use in the model forward.
        """
        
        # Set the template mode
        old_mode = self.prompt_template.mode
        self.prompt_template.set_mode(prompt_template_mode)

        formatted_prompt = self.prompt_template.get_prompt(prompt, model_context=model_context, suffix=infill_suffix,
                                                           system_prompt=system_prompt)
        
        # Reset template mode to previous mode
        self.prompt_template.set_mode(old_mode)

        return formatted_prompt
    

    def create_stopping_criteria(self,
                                 input_length: int,
                                 stopping_patterns: list[str] | tuple[str] | bool | None = None,
                                 parser: CodeParser | None = None
        ) -> tuple[StoppingCriteriaList | None, tuple[str] | list[str] | None]:
        """Create the stopping criteria to use from the `stopping_patterns`.

        Parameters
        ----------
        input_length : int
            Length of the input prompt.
        stopping_patterns : list[str] | tuple[str] | bool | None, optional
            List of words/patterns to stop the generation. Pass `True` to use the default 
            `EXTENDED_CODE_STOP_PATTERNS` patterns. If `None`, no early stopping is performed, by default None.
        parser: CodeParser | None, optional
            A parser to extract code from generated sequences. The `stopping_patterns` will be applied on the
            parsed sequences. This should be used with caution, as it was designed only for chat models that
            embed code in their output in natural language. The default is None, i.e. no parsing.

        Returns
        -------
        tuple[StoppingCriteriaList | None, tuple[str] | list[str] | None]
            Tuple containing the stopping criteria to use in the model forward, and the stopping patterns
            to use if we post-process the outputs.
        """

        # Possible early stopping
        if isinstance(stopping_patterns, list) or isinstance(stopping_patterns, tuple):
            stopping_criteria = stopping.TextPatternStopping(input_length, self.tokenizer, stopping_patterns,
                                                             self.extra_eos_tokens, parser)
            stopping_criteria = StoppingCriteriaList([stopping_criteria])
        elif isinstance(stopping_patterns, bool) and stopping_patterns:
            stopping_patterns = stopping.EXTENDED_CODE_STOP_PATTERNS
            stopping_criteria = stopping.TextPatternStopping(input_length, self.tokenizer, stopping_patterns,
                                                             self.extra_eos_tokens, parser)
            stopping_criteria = StoppingCriteriaList([stopping_criteria])
        else:
            stopping_patterns = None
            if len(self.extra_eos_tokens) == 0:
                stopping_criteria = None
            else:
                stopping_criteria = stopping.TextPatternStopping(input_length, self.tokenizer, stopping_patterns,
                                                                 self.extra_eos_tokens, parser)
                stopping_criteria = StoppingCriteriaList([stopping_criteria])

        return stopping_criteria, stopping_patterns
    

    def generate_text(
            self,
            prompt: str,
            model_context: str = '',
            infill_suffix: str = '',
            system_prompt: str = '',
            prompt_template_mode: str = 'default',
            max_new_tokens: int = 60,
            min_new_tokens: int = 5,
            do_sample: bool = True,
            top_k: int | None = 40,
            top_p: float | None = 0.90,
            temperature: float = 0.9,
            num_return_sequences: int = 1,
            batch_size: int | None = None,
            seed: int | None = None,
            stopping_patterns: tuple[str] | bool | None = None,
            parser: CodeParser | None = None,
            truncate_prompt_from_output: bool = True,
            post_process_output: bool = True,
            **kwargs
        ) -> str | list[str]:
        """Generate text according to `prompt` using the parameters specified.

        Prompt formatting parameters
        ----------------------------

        prompt : str
            The prompt to the model.
        model_context : str, optional
            An optional context forming the start of the model answer. For `generation` mode, this is simply
            appended to `prompt`. By default ''.
        infill_suffix : str, optional
            An optional suffix to form the prompt. This is ignored for all `prompt_template_mode` except
            `infill`, by default ''.
        system_prompt : str, optional
            An optional system prompt to append at the beginning for chat mode. This is ignored for all 
            `prompt_template_mode` except `chat`, by default ''.
        prompt_template_mode: str
            The template mode for formatting the prompt. One of `('default', 'generation', 'infill', 'chat')`.
            Note that changing this value may result in errors or inconsistent results as usually a model is
            optimized for only one given prompt format. By default 'default', which chooses the best mode for
            the current model.

        Generation parameters
        ---------------------
        
        max_new_tokens : int, optional
            How many new tokens to generate, by default 60.
        min_new_tokens : int, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
            force the model to generate an output, instead of immediately generating EOS, by default 5.
        do_sample : bool, optional
            Whether to introduce randomness in the generation, by default True.
        top_k : int | None, optional
            How many tokens with max probability to consider for random sampling, by default 50. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        top_p : float | None, optional
            The probability density covering the new tokens to consider for random sampling, by default 0.9. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.9. Passing 0 is equivalent to setting `do_sample=False`.
        num_return_sequences : int, optional
            How many sequences to generate according to the `prompt`, by default 1.
        batch_size : int | None, optional
            Max batch size for the model forward pass, in case `num_return_sequences` is large. If `None`, will
            try to determine the largest possible batch size that does not result in memory error. By default None.
        seed : int | None, optional
            An optional seed to force the generation to be reproducible.
        stopping_patterns: tuple[str] | bool | None, optional
            List of words/patterns to stop the generation. Pass `True` to use the default `EXTENDED_CODE_STOP_PATTERNS` patterns.
            If `None`, no early stopping is performed, by default None.
        parser: CodeParser | None, optional
            A parser to extract code from generated sequences. The final outputs will only consist of the parsed
            sequences if `post_process_output` is True. Also, the `stopping_patterns` will be applied on the
            parsed sequences. This should be used with caution, as it was designed only for chat models that
            embed code in their output in natural language. The default is None, i.e. no parsing.

        Output formatting parameters
        ----------------------------

        truncate_prompt_from_output : bool, optional
            Whether to remove the prompt from the model answer or not, by default True.
        post_process_output : bool, optional
            Whether to post-process the outputs, i.e. truncate according to the `stopping_patterns`. This is
            needed to correctly truncate all sequences if `num_return_sequences > 1`. By default True.

        Returns
        -------
        str | list[str]
            Str containing the generated sequence, or list[str] if `num_return_sequences` > 1.
        """
    
        if seed is not None:
            utils.set_all_seeds(seed)

        # Setting the temperature to 0 is equivalent to greedy search thus we explicitly set do_sample=False
        if temperature == 0:
            do_sample = False

        if num_return_sequences > 1 and not do_sample:
            warnings.warn(('You provided `do_sample=False` or `temperature=0`, i.e. greedy search generation, '
                           'but with `num_return_sequences>1`. All sequences will be identical with greedy search, '
                           'so we explicitly set `num_return_sequences=1`'))
            num_return_sequences = 1

        if do_sample:
            generation_kwargs = {'do_sample': do_sample, 'top_k': top_k, 'top_p': top_p, 'temperature': temperature}
        # If we are doing greedy search, do not pass arguments that alter the model output logits to `generate`
        else:
            generation_kwargs = {'do_sample': do_sample}

        # Prompt formatting
        formatted_prompt = self.format_prompt(prompt, model_context=model_context, infill_suffix=infill_suffix,
                                              system_prompt=system_prompt, prompt_template_mode=prompt_template_mode)
        
        # Prompt to reattach to output if `truncate_prompt_from_output` is False. This way we reattach the
        # prompt given directly by the user, and not the prompt formatted with potential keywords in all
        # but the most complicated cases
        if infill_suffix == '' and system_prompt == '':
            original_prompt = prompt + model_context
        else:
            original_prompt = formatted_prompt

        # Tokenize the prompt
        input = self.tokenizer.encode(formatted_prompt, return_tensors='pt')
        input_length = input.shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=self.input_device)

        # Create the stopping criteria
        stopping_criteria, stopping_patterns = self.create_stopping_criteria(input_length,
                                                                             stopping_patterns=stopping_patterns,
                                                                             parser=parser)

        # Suppress pad_token_id warning
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # Infer batch size if not given
        if batch_size is None:
            batch_size = self.infer_best_batch_size(input_length, max_new_tokens, num_return_sequences)

        # Anything larger than `num_return_sequences` is useless
        batch_size = min(batch_size, num_return_sequences)

        # This will lower the batch size if needed, in case of possible OOM. This allows to continue without crashing,
        # by reducing the batch size automatically
        first_output, batch_size = self.oom_safe_batch_generation(input, max_new_tokens=max_new_tokens,
                                                                  min_new_tokens=min_new_tokens,
                                                                  generation_kwargs=generation_kwargs,
                                                                  batch_size=batch_size,
                                                                  stopping_criteria=stopping_criteria,
                                                                  pad_token_id=pad_token_id, **kwargs)

        # If we require more sequences than the allowed batch size, we need to split the generation into
        # multiple passes
        if num_return_sequences > batch_size:
            batch_sizes = [batch_size]*(num_return_sequences // batch_size)
            remainder = num_return_sequences % batch_size
            if remainder != 0:
                batch_sizes += [remainder]
            assert sum(batch_sizes) == num_return_sequences
        else:
            batch_sizes = [num_return_sequences]

        generated_text = []

        for i, size in enumerate(batch_sizes):

            # Do not recompute the first batch of outputs
            if i == 0:
                outputs = first_output
            else:
                outputs = self.model.generate(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                              **generation_kwargs, num_return_sequences=size,
                                              stopping_criteria=stopping_criteria, pad_token_id=pad_token_id, **kwargs)
                
            # Truncate the prompt from the output
            truncated_outputs = outputs[:, input_length:]

            # Post-process the sequences according to stopping patterns and extra eos
            if post_process_output:
                generated_batch = stopping.post_process_sequences(truncated_outputs, self.tokenizer, stopping_patterns,
                                                                  self.extra_eos_tokens, parser)
            else:
                generated_batch = self.tokenizer.batch_decode(truncated_outputs, skip_special_tokens=True)
            
            # reattach the prompt if needed
            if not truncate_prompt_from_output:
                # generated_batch = [original_prompt + sequence for sequence in generated_batch]
                generated_batch = [formatted_prompt + sequence for sequence in generated_batch]
            
            generated_text += generated_batch

        # In this case return a str instead of list[str]
        if num_return_sequences == 1:
            generated_text = generated_text[0]

        return generated_text
    

    @utils.copy_docstring_and_signature(generate_text)
    def __call__(self, *args, **kwargs):
        return self.generate_text(*args, **kwargs)
    

    def infer_best_batch_size(self, input_size: int, max_new_tokens: int, num_return_sequences: int) -> int:
        """Try to infer the best (largest) possible batch size for the model given the current `input_size`,
        and `max_new_tokens`. By default, this function checks if a batch memory footprint estimation exists
        in the folder `memory_estimator`, and falls back to simple heuristics if this is not the case.

        Parameters
        ----------
        input_size : int
            The input length.
        max_new_tokens : int
            The number of tokens to generate.
        num_return_sequences : int
            The number of sequences to generate.
        Returns
        -------
        int
            Estimation of the largest possible batch size.
        """
    
        if not torch.cuda.is_available():
            memory = psutil.virtual_memory().total / 1024**3
        else:
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        available_memory = memory*0.9 - self.max_memory_footprint

        try:
            reference_file = os.path.join(utils.ROOT_FOLDER, 'memory_estimator', self.model_name, f'{self.dtype_category()}.json')
            batch_footprint = utils.load_json(reference_file)
            # Convert keys to int
            batch_footprint = {int(k): {int(k1):v1 for k1,v1 in batch_footprint[k].items()} for k in batch_footprint.keys()}
        # If no precise estimate exist, fall back to simple heuristics
        except FileNotFoundError:
            parameters = self.parameters_count()
            if parameters < 5:
                batch = int(available_memory // 0.5)
            elif parameters < 10:
                batch = int(available_memory // 1)
            elif parameters < 20:
                batch = int(available_memory // 2)
            else:
                batch = int(available_memory // 3)
            
            return max(batch, 1)

        # Find the reference input size immediately larger than the current input size. If none exist, take 
        # the largest and adapt with a coeff
        input_sizes = np.sort(list(batch_footprint.keys()))
        indices = np.nonzero(input_sizes >= input_size)
        if len(indices) == 0:
            ref_input_size = input_sizes[-1]
            input_size_coeff = input_size / ref_input_size
        else:
            ref_input_size = input_sizes[indices[0][0]]
            input_size_coeff = 1

        # Find the reference max new tokens immediately larger than the current max new tokens. If none exist, 
        # take the largest and adapt with a coeff
        max_tokens = np.sort(list(batch_footprint[ref_input_size].keys()))
        indices = np.nonzero(max_tokens >= max_new_tokens)
        if len(indices) == 0:
            ref_max_tokens = max_tokens[-1]
            max_tokens_coeff = max_new_tokens / ref_max_tokens
        else:
            ref_max_tokens = max_tokens[indices[0][0]]
            max_tokens_coeff = 1

        # Adapt the estimation with the coeffs if needed (they should usually be 1)
        ref_batch_footprint = batch_footprint[ref_input_size][ref_max_tokens] * input_size_coeff * max_tokens_coeff

        if ref_batch_footprint < 0:
            return num_return_sequences

        return int(available_memory // ref_batch_footprint)


    def oom_safe_batch_generation(self, input: torch.Tensor, max_new_tokens: int, min_new_tokens: int,
                                  generation_kwargs: dict, batch_size: int,
                                  stopping_criteria: StoppingCriteriaList | None, pad_token_id: int,
                                  **kwargs) -> tuple[torch.Tensor, int]:
        """Generate text by recursively recovering from possible memory errors (OOMs) by lowering the batch size.
        Note that it is not possible to retry immediately in the except block because the exception retains the
        tensors already allocated in the try block which causes an immediate new OOM
        (see https://github.com/pytorch/pytorch/issues/18853)
        """
        retry = False

        # Try generating result
        try:
            out = self.model.generate(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                      **generation_kwargs, num_return_sequences=batch_size,
                                      stopping_criteria=stopping_criteria, pad_token_id=pad_token_id, **kwargs)
        
        except RuntimeError as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                retry = True
            else:
                raise e

        if retry:
            if batch_size == 1:
                raise RuntimeError('Even a batch size of 1 causes an OOM. Cannot generate with current config.')
            new_batch_size = max(1, math.floor(batch_size*0.8))
            warnings.warn(f'Reducing batch size from {batch_size} to {new_batch_size} due to memory overflow (OOM).', RuntimeWarning)
            gc.collect()
            torch.cuda.empty_cache()
            return self.oom_safe_batch_generation(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                                  generation_kwargs=generation_kwargs, batch_size=new_batch_size,
                                                  stopping_criteria=stopping_criteria, pad_token_id=pad_token_id, **kwargs)
        else:
            return out, batch_size
        

    def parameters_count(self) -> float:
        """Return the (approximate) number of parameters of the current model, in billions.
        Note that shared parameters will be counted twice by this current function, thus it is only approximate.

        Returns
        -------
        float
            The number of parameters, in billions.
        """

        return sum(map(torch.numel, self.model.parameters())) / 1e9
    

    def set_prompt_template(self, template: GenericPromptTemplate):
        """Set the prompt template."""
        self.prompt_template = template


    def generate_conversation(
            self,
            prompt: str,
            system_prompt: str = '',
            conv_history: GenericConversation | None = None,
            max_new_tokens: int = 60,
            min_new_tokens: int = 5,
            do_sample: bool = True,
            top_k: int = 50,
            top_p: float = 0.90,
            temperature: float = 0.9,
            seed: int | None = None,
            truncate_if_conv_too_long: bool = True,
            **kwargs
    ) -> GenericConversation:
        """Generate a conversation turn between a user and the model, according to new user input `prompt`.

        Input parameters
        ----------------
        prompt : str
            The new prompt of the user to the model.
        system_prompt : str
            An optional system prompt to guide the style of the model answers.
        conv_history : GenericConversation | None
            An optional existing conversation object, representing the current dialogue between the user and
            the model.

        Generation parameters
        ---------------------

        max_new_tokens : int, optional
            How many new tokens to generate, by default 60.
        min_new_tokens : int, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
            force the model to generate an output, instead of immediately generating EOS, by default 5.
        do_sample : bool, optional
            Whether to introduce randomness in the generation, by default True.
        top_k : int | None, optional
            How many tokens with max probability to consider for random sampling, by default 50. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        top_p : float | None, optional
            The probability density covering the new tokens to consider for random sampling, by default 0.9. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.9. Passing 0 is equivalent to setting `do_sample=False`.
        seed : int | None, optional
            An optional seed to force the generation to be reproducible.
        truncate_if_conv_too_long : bool, optional
            Whether to truncate the conversation history if it becomes larger than the model maximum capacity,
            by default True.

        Returns
        -------
        GenericConversation
            A conversation object, with the dialogue history updated with the current turn.
        """

        if seed is not None:
            utils.set_all_seeds(seed)

        # Setting the temperature to 0 is equivalent to greedy search thus we explicitly set do_sample=False
        if temperature == 0:
            do_sample = False

        if do_sample:
            generation_kwargs = {'do_sample': do_sample, 'top_k': top_k, 'top_p': top_p, 'temperature': temperature}
        # If we are doing greedy search, do not pass arguments that alter the model output logits to `generate`
        else:
            generation_kwargs = {'do_sample': do_sample}

        # Check that the history is not empty
        if conv_history is None:
            conv_history = self.get_empty_conversation()

        # Set system prompt
        conv_history.set_system_prompt(system_prompt)

        # Add the prompt to the current conversation
        conv_history.append_user_message(prompt)

        # Generate and tokenize the full prompt
        full_prompt = conv_history.get_prompt()
        input = self.tokenizer.encode(full_prompt, return_tensors='pt')
        input_length = input.shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=self.input_device)

        # Create the stopping criteria in case the model has some extra eos tokens to process
        stopping_criteria, _ = self.create_stopping_criteria(input_length)

        # Suppress pad_token_id warning
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        outputs = self.model.generate(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                      **generation_kwargs, num_return_sequences=1,
                                      stopping_criteria=stopping_criteria, pad_token_id=pad_token_id, **kwargs)
                
        # Truncate the prompt from the output
        truncated_outputs = outputs[:, input_length:]

        # Post-process the sequences according to potential extra eos tokens
        response = stopping.post_process_sequences(truncated_outputs, self.tokenizer, stopping_patterns=None,
                                                   extra_eos_tokens=self.extra_eos_tokens)
        
        # Append output to the conv
        conv_history.append_model_message(response[0])

        return conv_history


    def get_empty_conversation(self) -> GenericConversation:
        """Return a new empty conversation with the template of the current model."""
        return get_conversation_template(self.model_name)
   



def expand_past_keys(past_key_values, batch_size):

    if batch_size <=1:
        return past_key_values
    
    new = []
    with torch.no_grad():
        for i in range(len(past_key_values)):
            new_ = []
            for j in range(len(past_key_values[i])):
                new_.append(past_key_values[i][j].repeat(batch_size, 1, 1))
            new.append(tuple(new_))

    return tuple(new)




