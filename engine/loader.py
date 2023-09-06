import warnings
import re
import math

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer

def _infer_model_size(model_name: str) -> float:
    """Return the number of parameters a model has from its name if it can be inferred from it. Raise a 
    ValueError otherwise.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    float
        The number of parameters of the model, in billions.
    """

    # The following regex matches any digits possibly separated with a dot ('.') which is immeditely
    # followed by a 'B' or 'M' to capture the model size following our model name convention. Parenthesis 
    # allow to capture given groups of the regex thanks to the match object .group() method.
    pattern = r'([0-9]+(?:\.[0-9]+)?)([BM])'

    match = re.search(pattern, model_name)
    if match:
        matched_number = match.group(1)
        matched_letter = match.group(2)
        # Model size in billion (B) of parameters
        model_size = float(matched_number) if matched_letter == 'B' else float(matched_number)/1e3
        return model_size
    else:
        raise ValueError('The model number of parameters cannot be inferred from its name.')
    

def _infer_model_sizes(name_mapping: dict[str, str]) -> dict[str, float]:
    """Infer the number of parameters of all model names (dict keys) and return them as {key: #params}.

    Parameters
    ----------
    name_mapping : dict[str, str]
        A dictionary whose keys are the model names.

    Returns
    -------
    dict[str, float]
        A mapping from names to number of parameters.
    """

    return {key: _infer_model_size(key) for key in name_mapping.keys()}

    

# Pretrained bloom models
BLOOM_MODELS_MAPPING = {
    'bloom-560M': 'bigscience/bloom-560m',
    'bloom-1.7B': 'bigscience/bloom-1b7',
    'bloom-3B': 'bigscience/bloom-3b',
    'bloom-7.1B':'bigscience/bloom-7b1',
    'bloom-176B': 'bigscience/bloom',
}
BLOOM_MODELS_DTYPES = {
    'bloom-560M': torch.float16,
    'bloom-1.7B': torch.float16,
    'bloom-3B': torch.float16,
    'bloom-7.1B':torch.float16,
    'bloom-176B': torch.bfloat16,
}
BLOOM_MODELS_PARAMS = _infer_model_sizes(BLOOM_MODELS_MAPPING)


# Pretrained Dialo-GPT models
DIALO_GPT_MODELS_MAPPING = {
    'dialo-gpt-small': 'microsoft/DialoGPT-small',
    'dialo-gpt-medium': 'microsoft/DialoGPT-medium',
    'dialo-gpt-large': 'microsoft/DialoGPT-large',
}
DIALO_GPT_MODELS_DTYPES = {
    'dialo-gpt-small': torch.float32,
    'dialo-gpt-medium': torch.float32,
    'dialo-gpt-large': torch.float32,
}
DIALO_GPT_MODELS_PARAMS = {
    'dialo-gpt-small': 125/1e3,
    'dialo-gpt-medium': 355/1e3,
    'dialo-gpt-large': 775/1e3,
}

# Pretrained StableLM models
STABLE_LM_MODELS_MAPPING = {
    'stable-lm-3B': 'stabilityai/stablelm-base-alpha-3b',
    'stable-lm-7B': 'stabilityai/stablelm-base-alpha-7b',
}
STABLE_LM_MODELS_DTYPES = {
    'stable-lm-3B': torch.float16,
    'stable-lm-7B': torch.float16,
}
STABLE_LM_MODELS_PARAMS = _infer_model_sizes(STABLE_LM_MODELS_MAPPING)

# Pretrained StarCoder models
STAR_CODER_MODELS_MAPPING = {
    'star-coder-base': 'bigcode/starcoderbase',
    # Star-coder-base further trained on Python
    'star-coder': 'bigcode/starcoder',
    # Star-coder-based further trained on English data
    'star-coder-plus': 'bigcode/starcoderplus',
}
STAR_CODER_MODELS_DTYPES = {
    'star-coder-base': torch.bfloat16,
    'star-coder': torch.bfloat16,
    'star-coder-plus': torch.bfloat16,
}
STAR_CODER_MODELS_PARAMS = {
    'star-coder-base': 15.5,
    'star-coder': 15.5,
    'star-coder-plus': 15.5,
}
STAR_CODER_ADDITIONAL_MODEL_KWARGS = {
    'star-coder-base': {'trust_remote_code': True},
}


# Pretrained Star-chat models
STAR_CHAT_MODELS_MAPPING = {
    'star-chat-alpha': 'HuggingFaceH4/starchat-alpha',
    'star-chat-beta': 'HuggingFaceH4/starchat-beta',
}
STAR_CHAT_MODELS_DTYPES = {
    'star-chat-alpha': torch.float16,
    'star-chat-beta': torch.bfloat16,
}
STAR_CHAT_MODELS_PARAMS = {
    'star-chat-alpha': 16,
    'star-chat-beta': 16,
}


# Pretrained GPT-2 models
GPT2_MODELS_MAPPING = {
    'gpt2-medium': 'gpt2-medium',
    'gpt2-large': 'gpt2-large',
    'gpt2-xl': 'gpt2-xl',
}
GPT2_MODELS_DTYPES = {
    'gpt2-medium': torch.float32,
    'gpt2-large': torch.float32,
    'gpt2-xl': torch.float32,
}
GPT2_MODELS_PARAMS = {
    'gpt2-medium': 355/1e3,
    'gpt2-large': 774/1e3,
    'gpt2-xl': 1.5,
}


# Pretrained GPT-J and GPT-Neo models
GPT_J_AND_NEO_MODELS_MAPPING = {
    'gpt-j-6B': 'EleutherAI/gpt-j-6B',
    'gpt-neo-125M': 'EleutherAI/gpt-neo-125m',
    'gpt-neo-1.3B': 'EleutherAI/gpt-neo-1.3B',
    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt-neoX-20B': 'EleutherAI/gpt-neox-20b',
}
GPT_J_AND_NEO_MODELS_DTYPES = {
    'gpt-j-6B': torch.float32,
    'gpt-neo-125M': torch.float32,
    'gpt-neo-1.3B': torch.float32,
    'gpt-neo-2.7B': torch.float32,
    'gpt-neoX-20B': torch.float16,
}
GPT_J_AND_NEO_MODELS_PARAMS = _infer_model_sizes(GPT_J_AND_NEO_MODELS_MAPPING)


# Pretrained OPT models
OPT_MODELS_MAPPING = {
    'opt-125M': 'facebook/opt-125m',
    'opt-350M': 'facebook/opt-350m',
    'opt-1.3B': 'facebook/opt-1.3b',
    'opt-2.7B': 'facebook/opt-2.7b',
    'opt-6.7B': 'facebook/opt-6.7b',
    'opt-13B': 'facebook/opt-13b',
    'opt-30B': 'facebook/opt-30b',
    'opt-66B': 'facebook/opt-66b',
}
OPT_MODELS_DTYPES = {
    'opt-125M': torch.float16,
    'opt-350M': torch.float16,
    'opt-1.3B': torch.float16,
    'opt-2.7B': torch.float16,
    'opt-6.7B': torch.float16,
    'opt-13B': torch.float16,
    'opt-30B': torch.float16,
    'opt-66B': torch.float16,
}
OPT_MODELS_PARAMS = _infer_model_sizes(OPT_MODELS_MAPPING)


# Pretrained CodeGEN models
CODEGEN_MODELS_MAPPING = {
    'codegen-350M': 'Salesforce/codegen-350M-mono',
    'codegen-2B': 'Salesforce/codegen-2B-mono',
    'codegen-6B': 'Salesforce/codegen-6B-mono',
    'codegen-16B': 'Salesforce/codegen-16B-mono',
}
CODEGEN_MODELS_DTYPES = {
    'codegen-350M': torch.float16,
    'codegen-2B': torch.float16,
    'codegen-6B': torch.float16,
    'codegen-16B': torch.float16,
}
CODEGEN_MODELS_PARAMS = _infer_model_sizes(CODEGEN_MODELS_MAPPING)


# Pretrained CodeGEN2 models
CODEGEN2_MODELS_MAPPING = {
    'codegen2-1B': 'Salesforce/codegen2-1B',
    'codegen2-3.7B': 'Salesforce/codegen2-3_7B',
    'codegen2-7B': 'Salesforce/codegen2-7B',
    'codegen2-16B': 'Salesforce/codegen2-16B',
    'codegen25-7B': 'Salesforce/codegen25-7B-mono',
    'codegen25-7B-instruct': 'Salesforce/codegen25-7b-instruct',
}
CODEGEN2_MODELS_DTYPES = {
    'codegen2-1B': torch.float16,
    'codegen2-3.7B': torch.float16,
    'codegen2-7B': torch.float16,
    'codegen2-16B': torch.float16,
    'codegen25-7B': torch.float16,
    'codegen25-7B-instruct': torch.float16,
}
CODEGEN2_MODELS_PARAMS = _infer_model_sizes(CODEGEN2_MODELS_MAPPING)
CODEGEN2_ADDITIONAL_MODEL_KWARGS = {
    'codegen2-1B': {'trust_remote_code': True, 'revision': 'main'},
    'codegen2-3.7B': {'trust_remote_code': True, 'revision': 'main'},
    'codegen2-7B': {'trust_remote_code': True, 'revision': 'main'},
    'codegen2-16B': {'trust_remote_code': True, 'revision': 'main'},
}
CODEGEN2_ADDITIONAL_TOKENIZER_KWARGS = {
    'codegen25-7B': {'trust_remote_code': True},
    'codegen25-7B-instruct': {'trust_remote_code': True},
}


# Pretrained Vicuna (1.3) models
VICUNA_MODELS_MAPPING = {
    'vicuna-7B': 'lmsys/vicuna-7b-v1.3',
    'vicuna-13B': 'lmsys/vicuna-13b-v1.3',
}
VICUNA_MODELS_DTYPES = {
    'vicuna-7B': torch.float16,
    'vicuna-13B': torch.float16,
}
VICUNA_MODELS_PARAMS = _infer_model_sizes(VICUNA_MODELS_MAPPING)
# Fast tokenizers and non-legacy behaviour are bugged in current transformers and tokenizers versions
# TODO: may need to be changed in future versions if they correct the bug
VICUNA_ADDITIONAL_TOKENIZER_KWARGS = {
    'vicuna-7B': {'use_fast_tokenizer': False, 'legacy':True},
    'vicuna-13B': {'use_fast_tokenizer': False, 'legacy':True},
}

# Pretrained llama-2 models
LLAMA2_MODELS_MAPPING = {
    'llama2-7B': 'meta-llama/Llama-2-7b-hf',
    'llama2-13B': 'meta-llama/Llama-2-13b-hf',
    'llama2-70B': 'meta-llama/Llama-2-70b-hf',
    'llama2-7B-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13B-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2-70B-chat': 'meta-llama/Llama-2-70b-chat-hf',
}
LLAMA2_MODELS_DTYPES = {
    'llama2-7B': torch.float16,
    'llama2-13B': torch.float16,
    'llama2-70B': torch.float16,
    'llama2-7B-chat': torch.float16,
    'llama2-13B-chat': torch.float16,
    'llama2-70B-chat': torch.float16,
}
LLAMA2_MODELS_PARAMS = _infer_model_sizes(LLAMA2_MODELS_MAPPING)
# Fast tokenizers and non-legacy behaviour are bugged in current transformers and tokenizers versions
# TODO: may need to be changed in future versions if they correct the bug
LLAMA2_ADDITIONAL_TOKENIZER_KWARGS = {
    'llama2-7B': {'use_fast_tokenizer': False, 'legacy':True},
    'llama2-13B': {'use_fast_tokenizer': False, 'legacy':True},
    'llama2-70B': {'use_fast_tokenizer': False, 'legacy':True},
    'llama2-7B-chat': {'use_fast_tokenizer': False, 'legacy':True},
    'llama2-13B-chat': {'use_fast_tokenizer': False, 'legacy':True},
    'llama2-70B-chat': {'use_fast_tokenizer': False, 'legacy':True},
}


# Decoder-based models
DECODER_MODELS_MAPPING = {
    **BLOOM_MODELS_MAPPING,
    **DIALO_GPT_MODELS_MAPPING,
    **STABLE_LM_MODELS_MAPPING,
    **STAR_CODER_MODELS_MAPPING,
    **STAR_CHAT_MODELS_MAPPING,
    **GPT2_MODELS_MAPPING,
    **GPT_J_AND_NEO_MODELS_MAPPING,
    **OPT_MODELS_MAPPING,
    **CODEGEN_MODELS_MAPPING,
    **CODEGEN2_MODELS_MAPPING,
    **VICUNA_MODELS_MAPPING,
    **LLAMA2_MODELS_MAPPING,
}
DECODER_MODELS_DTYPES_MAPPING = {
    **BLOOM_MODELS_DTYPES,
    **DIALO_GPT_MODELS_DTYPES,
    **STABLE_LM_MODELS_DTYPES,
    **STAR_CODER_MODELS_DTYPES,
    **STAR_CHAT_MODELS_DTYPES,
    **GPT2_MODELS_DTYPES,
    **GPT_J_AND_NEO_MODELS_DTYPES,
    **OPT_MODELS_DTYPES,
    **CODEGEN_MODELS_DTYPES,
    **CODEGEN2_MODELS_DTYPES,
    **VICUNA_MODELS_DTYPES,
    **LLAMA2_MODELS_DTYPES,
}
DECODER_MODELS_PARAMS_MAPPING = {
    **BLOOM_MODELS_PARAMS,
    **DIALO_GPT_MODELS_PARAMS,
    **STABLE_LM_MODELS_PARAMS,
    **STAR_CODER_MODELS_PARAMS,
    **STAR_CHAT_MODELS_PARAMS,
    **GPT2_MODELS_PARAMS,
    **GPT_J_AND_NEO_MODELS_PARAMS,
    **OPT_MODELS_PARAMS,
    **CODEGEN_MODELS_PARAMS,
    **CODEGEN2_MODELS_PARAMS,
    **VICUNA_MODELS_PARAMS,
    **LLAMA2_MODELS_PARAMS,
}
DECODER_ADDITIONAL_MODEL_KWARGS_MAPPING = {
    **STAR_CODER_ADDITIONAL_MODEL_KWARGS,
    **CODEGEN2_ADDITIONAL_MODEL_KWARGS,
}
DECODER_ADDITIONAL_TOKENIZER_KWARGS_MAPPING = {
    **CODEGEN2_ADDITIONAL_TOKENIZER_KWARGS,
    **VICUNA_ADDITIONAL_TOKENIZER_KWARGS,
    **LLAMA2_ADDITIONAL_TOKENIZER_KWARGS,
}



# Pretrained BERT models
BERT_MODELS_MAPPING = {
    'bert-base-uncased': 'bert-base-uncased',
    'bert-large-uncased': 'bert-large-uncased',
    'bert-base-cased': 'bert-base-cased',
    'bert-large-cased': 'bert-large-cased',
}


# Pretrained RoBERTa models
ROBERTA_MODELS_MAPPING = {
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-large',
}


# Encoder-based models
ENCODER_MODELS_MAPPING = {
    **BERT_MODELS_MAPPING,
    **ROBERTA_MODELS_MAPPING,
}



# Pretrained BART models
BART_MODELS_MAPPING = {
    'bart-base': 'facebook/bart-base',
    'bart-large': 'facebook/bart-large',
}


# Pretrained T5 models
T5_MODELS_MAPPING = {
    't5-small': 't5-small',
    't5-base': 't5-base',
    't5-large': 't5-large',
    't5-3B': 't5-3b',
    't5-11B': 't5-11b',
}


# Pretrained FLAN-T5 models
FLAN_T5_MODELS_MAPPING = {
    'flan-t5-small': 'google/flan-t5-small',
    'flan-t5-base': 'google/flan-t5-base',
    'flan-t5-large': 'google/flan-t5-large',
    'flan-t5-xl': 'google/flan-t5-xl',
    'flan-t5-xxl': 'google/flan-t5-xxl',
}


# Full transformer-based (encoder + decoder) models
TRANSFORMER_MODELS_MAPPING = {
    **BART_MODELS_MAPPING,
    **T5_MODELS_MAPPING,
    **FLAN_T5_MODELS_MAPPING,
}





# All models mapping
ALL_MODELS_MAPPING = {
    **DECODER_MODELS_MAPPING,
    **ENCODER_MODELS_MAPPING, 
    **TRANSFORMER_MODELS_MAPPING,
}
ALL_MODELS_DTYPES_MAPPING = {
    **DECODER_MODELS_DTYPES_MAPPING,
}
ALL_MODELS_PARAMS_MAPPING = {
    **DECODER_MODELS_PARAMS_MAPPING,
}
ALL_MODELS_ADDITIONAL_MODEL_KWARGS_MAPPING = {
    **DECODER_ADDITIONAL_MODEL_KWARGS_MAPPING,
}
ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS_MAPPING = {
    **DECODER_ADDITIONAL_TOKENIZER_KWARGS_MAPPING,
}

# Summarize all supported model names
ALLOWED_MODELS = tuple(ALL_MODELS_MAPPING.keys())

ALLOWED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)



def get_model_params(model_name: str) -> float:
    """Return the approximate number of params of the model, in billions.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    float
        The number of parameters.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    return ALL_MODELS_PARAMS_MAPPING[model_name]


def get_model_dtype(model_name: str) -> torch.dtype:
    """Return the default dtype used by the model.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    torch.dtype
        The default dtype.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    return ALL_MODELS_DTYPES_MAPPING[model_name]


def estimate_model_gpu_footprint(model_name, quantization_8bits: bool = False, quantization_4bits: bool = False,
                                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8,
                                 max_fraction_gpus: float = 0.8) -> tuple[int, dict]:
    """Estimate the minimum number of gpus needed to perform inference with a model, given the maximum gpu memory
    proportion `max_fraction_gpu_0` and `max_fraction_gpus` that we allow for the model. This relies on
    simple heuristics. This also computes the corresponding `memory_map` to use when creating a `device_map`.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 8 bits mode, by default False.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8.

    Returns
    -------
    tuple[int, dict]
        Tuple containing the minimum number of gpus needed, the `memory_map`, i.e. a dictionary mapping each gpu
        needed to the maximum size reserved by the model for this gpu.
    """

    if max_fraction_gpu_0 < 0 or max_fraction_gpus < 0:
        raise ValueError('The maximum fraction of gpu memory to use cannot be negative.')
    
    if max_fraction_gpu_0 > 0.85 or max_fraction_gpus > 0.85:
        raise ValueError(('The maximum fraction of gpu memory to use cannot be larger than 0.85 because some '
                         'memory need to stay free for the forward pass and other computations.'))
    
    # Silently use 4bits when both are True
    if quantization_4bits and quantization_8bits:
        quantization_8bits = False

    # If not provided take the default one
    if dtype is None:
        dtype = get_model_dtype(model_name)

    if quantization_4bits:
        size_multiplier = 1/2
    elif quantization_8bits:
        size_multiplier = 1
    elif (dtype == torch.float16) or (dtype == torch.bfloat16):
        size_multiplier = 2
    else:
        size_multiplier = 4

    # Estimate of the memory size of the model
    rough_model_size_estimate = ALL_MODELS_PARAMS_MAPPING[model_name] * size_multiplier
    
    # We assume that we always have identical gpus when using multiple gpus
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Say we only have access to a portion of that memory for our model
    gpu_0_available_memory = max_fraction_gpu_0 * gpu_memory
    gpus_available_memory = max_fraction_gpus * gpu_memory

    # Heuristic: if the remainder is smaller than 2% of gpu_memory, do not add a gpu 
    if rough_model_size_estimate <= gpu_0_available_memory + 0.02 * gpu_memory:
        return 1, None
    
    else:
        max_memory_map = {0: f'{math.ceil(gpu_0_available_memory)}GiB'}
        to_fit_on_other_gpus = rough_model_size_estimate - gpu_0_available_memory
        additional_gpus_needed = int(to_fit_on_other_gpus // gpus_available_memory)

        # Heuristic: if the remainder is smaller than 2% of each gpu_memory, do not add a gpu and distill
        # the small excess between existing gpus
        if to_fit_on_other_gpus % gpus_available_memory >= (0.02 * gpu_memory) * additional_gpus_needed:
            additional_gpus_needed += 1
            available_gpu_size = math.ceil(gpus_available_memory)
        else:
            # Add the 2% to the gpus size requirements
            available_gpu_size = math.ceil((max_fraction_gpus + 0.02) * gpu_memory)

        gpu_needed = 1 + additional_gpus_needed
        for i in range(1, gpu_needed):
            max_memory_map[i] = f'{available_gpu_size}GiB'

        return gpu_needed, max_memory_map



def load_model(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
               dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
               device_map: dict | None = None, gpu_rank: int = 0):
    """Load one of the supported pretrained model.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_4bits`.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_8bits`.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    device_map : dict | None, optional
        The device map to decide how to split the model between available devices, by default None. If not
        provided, the model dispatch to GPU(s) is made according to `max_fraction_gpu_0` and `max_fraction_gpus`
        in such a way to use the smallest number of gpus that respect these two values.
    gpu_rank : int, optional
        The gpu rank on which to put the model if it can fit on a single gpu. This is ignored if `device_map`
        is provided. By default 0.

    Returns
    -------
        The model.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    # Set the dtype if not provided
    if dtype is None:
        dtype = ALL_MODELS_DTYPES_MAPPING[model_name]

    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f'The dtype must be one of {*ALLOWED_DTYPES,}.')
    
    if quantization_8bits and quantization_4bits:
        raise ValueError(('You cannot load a model with both `quantization_8bits` and `quantization_4bits`. '
                         'Please choose one'))
    
    if isinstance(device_map, str):
        raise ValueError(("You cannot provide the `device_map` as a string. Please use the `max_fraction_gpu_0` "
                         "and `max_fraction_gpus` arguments to mimic `device_map='balanced'` or "
                         "`device_map='balanced_low_0'` in a more coherent way."))
    
    # torch.float16 is not supported on cpu
    if not torch.cuda.is_available() and dtype != torch.float32:
        dtype = torch.float32
    
    # Override quantization if we don't have access to GPUs
    if not torch.cuda.is_available() and (quantization_8bits or quantization_4bits):
        quantization_4bits = False
        quantization_8bits = False
        warnings.warn('There are no GPUs available. The model will NOT be quantized.', RuntimeWarning)

    # Flag to know if the model is quantized
    quantization = quantization_8bits or quantization_4bits

    # Override dtype if we quantize the model as only float16 is acceptable for quantization
    dtype = torch.float16 if quantization else dtype

    # Add possible additional kwargs
    if model_name in ALL_MODELS_ADDITIONAL_MODEL_KWARGS_MAPPING.keys():
        additional_kwargs = ALL_MODELS_ADDITIONAL_MODEL_KWARGS_MAPPING[model_name]
    else:
        additional_kwargs = {}


    # Flag that will be set to True if we don't even need a device_map and can just put the model on one gpu
    only_move_to_one_gpu = False
    
    # Automatically find the best device_map depending on the model size and gpu size.
    # Try to minimize the number of gpus to use because using more will slow inference (but allow larger
    # batch size -> hard trade-off to find). Indeed, the parallelism of device_map is naive and gpus are only
    # used sequentially
    if (device_map is None) and torch.cuda.is_available():
    
        min_gpu_needed, max_memory_map = estimate_model_gpu_footprint(model_name, quantization_8bits=quantization_8bits,
                                                                      quantization_4bits=quantization_4bits, dtype=dtype,
                                                                      max_fraction_gpu_0=max_fraction_gpu_0,
                                                                      max_fraction_gpus=max_fraction_gpus)
        gpu_number = torch.cuda.device_count()

        if min_gpu_needed > gpu_number:
            raise RuntimeError(("The model seems too big for the gpu resources you have. To offload to the cpu as well, "
                               "explicitly pass a `device_map`, e.g device_map='balanced'."))
        
        # In this case we don't need a device_map, we just move the model to the 1st gpu. Most models are 
        # relatively small and should fall on this category.
        if min_gpu_needed == 1:
            only_move_to_one_gpu = True
        # In this case, we need more than 1 gpu so we create a device_map between different gpus. However, 
        # we minimize the number of gpus used with the max_memory arg instead of naively using device_map='balanced'
        # between all gpus, because the parallelism is not optimized and thus using a lot of gpus is not efficient
        # if not needed
        else:
            additional_kwargs['max_memory'] = max_memory_map
            # Providing 'balanced' dispatch correctly with respect to the max_memory_map we provide
            device_map = 'balanced'

    
    # Initiate different model types depending on architecture
    if model_name in DECODER_MODELS_MAPPING.keys():
        model = AutoModelForCausalLM.from_pretrained(DECODER_MODELS_MAPPING[model_name], device_map=device_map,
                                                    torch_dtype=dtype, load_in_8bit=quantization_8bits,
                                                    load_in_4bit=quantization_4bits, low_cpu_mem_usage=True,
                                                    **additional_kwargs)
    elif model_name in ENCODER_MODELS_MAPPING.keys():
        model = AutoModelForMaskedLM.from_pretrained(ENCODER_MODELS_MAPPING[model_name], device_map=device_map,
                                                    torch_dtype=dtype, load_in_8bit=quantization_8bits,
                                                    load_in_4bit=quantization_4bits, low_cpu_mem_usage=True,
                                                    **additional_kwargs)
    elif model_name in TRANSFORMER_MODELS_MAPPING.keys():
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSFORMER_MODELS_MAPPING[model_name], device_map=device_map,
                                                      torch_dtype=dtype, load_in_8bit=quantization_8bits,
                                                      load_in_4bit=quantization_4bits, low_cpu_mem_usage=True,
                                                      **additional_kwargs)
    
    # If the flag is active we directly put our model on one gpu without using any device_map (this is 
    # more efficient). But if the model is quantized, this is already done automatically because quantization
    # happen only on gpu
    if only_move_to_one_gpu and not quantization:
        # This operation is in-place for nn.Module
        model.cuda(gpu_rank)

    # For some reason bettertransformer is supported for codegen2 models but makes them crash during the forward
    if not ('codegen2-' in model_name):
        # Convert to better transformer to use Pytorch optimizations if supported by the model
        try:
            model = model.to_bettertransformer()
        except:
            pass
        
    model.eval()

    return model


def load_tokenizer(model_name: str):
    """Load a pretrained tokenizer corresponding to one of the supported models.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
        The tokenizer.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.') 
    
    if model_name in ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS_MAPPING.keys():
        additional_kwargs = ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS_MAPPING[model_name]
    else:
        additional_kwargs = {}
    
    tokenizer = AutoTokenizer.from_pretrained(ALL_MODELS_MAPPING[model_name], **additional_kwargs)

    return tokenizer


def load_model_and_tokenizer(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
                             dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8,
                             max_fraction_gpus: float = 0.8, device_map: dict | None = None,
                             gpu_rank: int = 0) -> tuple:
    """Load both a model and corresponding tokenizer.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_4bits`.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_8bits`.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    device_map : dict | None, optional
        The device map to decide how to split the model between available devices, by default None. If not
        provided, the model dispatch to GPU(s) is made according to `max_fraction_gpu_0` and `max_fraction_gpus`
        in such a way to use the smallest number of gpus that respect these two values.
    gpu_rank : int, optional
        The gpu rank on which to put the model if it can fit on a single gpu. This is ignored if `device_map`
        is provided. By default 0.

    Returns
    -------
    tuple
        The model and tokenizer.
    """

    return (load_model(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits,
                       dtype=dtype, max_fraction_gpu_0=max_fraction_gpu_0, max_fraction_gpus=max_fraction_gpus,
                       device_map=device_map, gpu_rank=gpu_rank),
            load_tokenizer(model_name))

