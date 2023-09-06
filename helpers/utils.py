import os
import json
import time
import random
import textwrap
import multiprocessing as mp
from typing import Callable, TypeVar, ParamSpec

import numpy as np


P = ParamSpec("P")
T = TypeVar("T")

# Path to the root of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))

# Path to the data folder
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

# Path to the results folder
RESULTS_FOLDER = os.path.join(ROOT_FOLDER, 'results')


# Most frequent text/data file extensions
FREQUENT_EXTENSIONS = (
    'json',
    'jsonl',
    'txt',
    'csv'
)


def set_all_seeds(seed: int):
    """Set seed for all random number generators (random, numpy and torch).

    Parameters
    ----------
    seed : int
        The seed.
    """

    # We import here to avoid having to use torch as a dependency of this utils module if this function is
    # not needed 
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_output(predictions: list[str]) -> str:
    """Format a list of strings corresponding to model predictions into a single string.

    Parameters
    ----------
    predictions : list[str]
        The model predictions.

    Returns
    -------
    str
        Formatted string.
    """

    if len(predictions) == 1:
        return predictions[0]
    else:
        out = f''
        for i, pred in enumerate(predictions):
            out += f'Sequence {i+1}:\n{pred}'
            if i != len(predictions)-1:
                out += '\n\n'
        return out
    

def get_hf_token(token_file: str = '.hf_token.txt') -> str:
    """Return the huggingface identification token stored in `token_file`.

    Parameters
    ----------
    token_file : str, optional
        File where the token is stored, by default '.hf_token.txt'

    Returns
    -------
    str
        The huggingface identification token.
    """

    if ROOT_FOLDER not in token_file:
        token_file = os.path.join(ROOT_FOLDER, token_file)

    try:
        with open(token_file, 'r') as file:
            # Read lines and remove whitespaces
            token = file.readline().strip()
    
    except FileNotFoundError:
        raise ValueError('The file you provided for your token does not exist.')

    return token


def validate_filename(filename: str, extension: str = 'json') -> str:
    """Format and check the validity of a filename and its extension. Create the path if needed, and 
    add/manipulate the extension if needed.

    Parameters
    ----------
    filename : str
        The filename to check for.
    extension : str, optional
        The required extension for the filename, by default 'json'

    Returns
    -------
    str
        The filename, reformated if needed.
    """

    # Extensions are always lowercase
    extension = extension.lower()
    # Remove dots in extension if any
    extension = extension.replace('.', '')

    dirname, basename = os.path.split(filename)

    # Check that the extension and basename are correct
    if basename == '':
        raise ValueError('The basename cannot be empty')
    
    split_on_dots = basename.split('.')

    # In this case add the extension at the end
    if len(split_on_dots) == 1:
        basename += '.' + extension
    # In this case there may be an extension, and we check that it is the correct one and change it if needed
    else:
        # The extension is correct
        if split_on_dots[-1] == extension:
            pass
        # There is a frequent extension, but not the correct one -> we change it
        elif split_on_dots[-1] in FREQUENT_EXTENSIONS:
            basename = '.'.join(split_on_dots[0:-1]) + '.' + extension
        # We did not detect any extension -> just add it at the end
        else:
            basename = '.'.join(split_on_dots) + '.' + extension

    # Check that the given path goes through the project repository
    dirname = os.path.abspath(dirname)
    if not (dirname.startswith(ROOT_FOLDER + os.sep) or dirname == ROOT_FOLDER):
        raise ValueError('The path you provided is outside the project repository.')

    # Make sure the path exists, and creates it if this is not the case
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return os.path.join(dirname, basename)


def save_json(dictionary: dict, filename: str):
    """
    Save a dictionary to disk as a json file.

    Parameters
    ----------
    dictionary : dict
        The dictionary to save.
    filename : str
        Filename to save the file.
    """
    
    filename = validate_filename(filename, extension='json')
    
    with open(filename, 'w') as fp:
        json.dump(dictionary, fp, indent='\t')
        
        
def save_jsonl(dictionaries: list[dict], filename: str, append: bool = False):
    """Save a list of dictionaries to a jsonl file.

    Parameters
    ----------
    dictionaries : list[dict]
        The list of dictionaries to save.
    filename : str
        Filename to save the file.
    append : bool
        Whether to append at the end of the file or create a new one, default to False.
    """

    filename = validate_filename(filename, extension='jsonl')

    mode = 'a' if append else 'w'

    with open(filename, mode) as fp:
        for dic in dictionaries:
            fp.write(json.dumps(dic) + '\n')


def save_txt(strings: list[str] | str, filename: str, separator: str = '\n'):
    """Write a list of strings to txt file, separated by `separator` character.

    Parameters
    ----------
    strings : list[str] | str
        String(s) to save.
    filename : str
        Filename to save the file.
    separator : str, optional
        Character used to separate each list element, by default '\n'.
    """

    if isinstance(strings, str):
        strings = [strings]

    filename = validate_filename(filename, extension='txt')

    with open(filename, 'w') as fp:
        for i, s in enumerate(strings):
            fp.write(s)
            # Only add separator if s is not the last item
            if i < len(strings) - 1:
                fp.write(separator)


def load_json(filename: str) -> dict:
    """
    Load a json file and return a dictionary.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    data : dict
        The dictionary representing the file.

    """
    
    with open(filename, 'r') as fp:
        data = json.load(fp)

    return data


def load_jsonl(filename: str) -> list[dict]:
    """Load a jsonl file as a list of dictionaries.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    list[dict]
        The list of dictionaries.
    """

    dictionaries = []

    with open(filename, 'r') as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                # yield json.loads(line)
                dictionaries.append(json.loads(line))

    return dictionaries


def load_txt(filename: str, separator: str = '\n') -> list[str]:
    """Load a txt file into a list of strings.

    Parameters
    ----------
    filename : str
        Filename to load.
    separator : str, optional
        The separator used to separate the file, by default '\n'

    Returns
    -------
    list[str]
        The content of the file.
    """

    with open(filename, 'r') as fp:
        file = fp.read()

    strings = file.split(separator)

    return strings


def find_rank_of_subprocess_inside_the_pool():
    """Find the rank of the current subprocess inside the pool that was launched either by 
    multiprocessing.Pool() or concurrent.futures.ProcessPoolExecutor(), starting with rank 0.
    If called from the main process, return -1.
    Note that this is a bit hacky but work correctly because both methods provide the rank of the subprocesses
    inside the subprocesses name as 'SpawnPoolWorker-RANK' or 'SpawnProcess-RANK' respectively.
    """

    process = mp.current_process()

    if process.name == 'MainProcess':
        rank = -1
    elif isinstance(process, mp.context.SpawnProcess) or isinstance(process, mp.context.ForkProcess):
        # Provide rank starting at 0 instead of 1
        try:
            rank = int(process.name.rsplit('-', 1)[1]) - 1
        except ValueError:
            raise RuntimeError('Cannot retrieve the rank of the current subprocess.')
    else:
        raise RuntimeError('The type of the running process is unknown.')
        
    return rank


def set_cuda_visible_device(gpu_rank: int | list[int]):
    """Set cuda visible devices to `gpu_rank` only.

    Parameters
    ----------
    gpu_rank : int | list[int]
        The GPUs we want to be visible.
    """

    if type(gpu_rank) == int:
        gpu_rank = [gpu_rank]

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_rank)


def set_cuda_visible_device_of_subprocess():
    """Set the cuda visible device of a subprocess inside a pool to only the gpu with same
    rank as the subprocess.
    """

    gpu_rank = find_rank_of_subprocess_inside_the_pool()
    set_cuda_visible_device(gpu_rank)


def copy_docstring_and_signature(copied_func: Callable[P, T]):
    """Decorator that copies the docstring and signature of another function.
    Note: the type hints are absolutely necessary for VScode to properly show the signature and docstring
    of the new function. There is some black magic in how VScode shows the docstrings because simply
    updating the __doc__ property does not work...
    
    Parameters
    ----------
    copied_func : Callable[P, T]
        The function from which to copy docstring and signature.
    """

    def wrapper(original_func: Callable[P, T]) -> Callable[P, T]:
        original_func.__doc__ = copied_func.__doc__
        return original_func
    
    return wrapper


def duplicate_function_for_gpu_dispatch(func: Callable[P, T]) -> Callable[P, T]:
    """This decorator is used for convenience when working with `dispatch_jobs` to run a function using
    an arbitrary number of gpus in parallel. It creates a function identical to the decorated function in the
    GLOBAL scope of the `utils` module, with `_gpu_dispatch` concatenated to the name. The only difference with
    the original function is that the first argument is a `list[int] | int`, representing the gpus that are set
    to be visible in the `CUDA_VISIBLE_DEVICES` env variable. This is needed to start a new process and correctly
    set the visible gpus for the process. The function needs to be global because otherwise it's not pickable, and
    we cannot start a `mp.Process` with `spawn` context and a non-pickable function. This is why we do not simply
    define the extended function inside the `dispatch_jobs` function. Since the decorator is
    executed when the file is first read, the function will correctly exist in the global scope. The original
    function `func` is also added to the global scope of the `utils` module so that it can be called later.

    NOTE: The function will exist only in the global scope of the current module, i.e. `utils`. If you import 
    the module elsewhere, e.g. `from helpers import utils`, then you will need to call the function as
    `utils.xxx_gpu_dispatch` as it lives in the globals of `utils`.

    Example:

    ```python
    @duplicate_function_for_gpu_dispatch
    def test(a: int):
        return a

    # original function
    test(10)
    >>> 10

    gpus = [0,1]
    # global created by the decorator
    test_gpu_dispatch(gpus, 10)
    >>> 10

    # Missing first `gpus` argument, so it will complain that the original function miss the first argument
    test_gpu_dispatch(10)
    >>> TypeError: test() missing 1 required positional argument: 'a'
    ```

    Returns
    -------
    Callable[P, T]
        The original function `func`. The original is not modified and is returned as is, we only create
        a new one in addition.
    """

    name = func.__name__
    correct_name = f'{name}_gpu_dispatch'

    if name in globals().keys():
        raise RuntimeError(("Cannot duplicate the function, because it would overwrite an existing global",
                            f" variable with the name '{name}'"))
    if correct_name in globals().keys():
        raise RuntimeError(("Cannot duplicate the function, because it would overwrite an existing global",
                            f" variable with the name '{correct_name}'"))
    
    # Add the function to current globals() so that it can be called later on
    globals()[name] = func
    
    block_str = f"""
                global {correct_name}
                def {correct_name}(gpus: list[int] | int, *args: P.args, **kwargs: P.kwargs) -> T:
                    set_cuda_visible_device(gpus)
                    return {name}(*args, **kwargs)
                """
    # remove indentation before each new line (triple quotes don't respect indentation level)
    block_str = textwrap.dedent(block_str)

    # Execute the string to actually define the new function in global scope
    exec(block_str)
    
    return func


def dispatch_jobs(model_names: list[str], model_footprints: list[int], num_gpus: int,
                  target_func: Callable[..., None], func_args: tuple | None = None,
                  func_kwargs: dict | None = None):
    """Run all jobs that need more than one gpu in parallel. Since the number of gpus needed by the models
    is variable, we cannot simply use a Pool of workers and map `target_func` to the Pool, or create processes and
    then ".join" them. To overcome this limitation, we use an infinite while loop that is refreshed by the main
    process every 10s. The dispatch of models to gpus is very naive: as soon as enough gpus are available to
    run the job that requires the less gpu, we launch it. Thus the gpu efficiency may not be the best possible.
    However, this would be extremely hard to improve on this simple strategy, especially since we do not know
    the runtime of each job.

    ## Caution - Insidious issues
    Some imports on the launching script (the script in which this function is called) may somehow cause 
    `torch.cuda` to be initialized before seeing the `CUDA_VISIBLE_DEVICES` that were set in the child processes.
    This is especially the case of the following: `from transformers import PreTrainedModel`. Even doing
    ```python
    import transformers
    def test(a: transformers.PreTrainedModel):
        pass
    ``` 
    i.e. using `PreTrainedModel` as a type hint in some file that was imported causes the issue of initializing
    `torch.cuda` before seeing the env variable. This creates bugs that are very hard to understand and follow,
    so be careful about what you actually import in the launching script. For this reason, all `PreTrainedModel`
    references were suppressed of the files in this project.


    Parameters
    ----------
    model_names : list[str]
        The list of model names to map to the `target_func`. This should be the first argument of `target_func`.
    model_footprints : list[int]
        The number of gpus required for each call to `target_func` (corresponding to each `model_names`). 
    num_gpus : int
        The total number of gpus available to dispatch the jobs.
    target_func : Callable[..., None]
        The function to call with each `model_names`. The first argument of the function should be a str
        representing the model name. The function should also be decorated with `@duplicate_function_for_gpu_dispatch`.
        If it returns a value, this value will be ignored.
    func_args : tuple | None, optional
        A tuple of additional arguments to pass to `target_func`. Calls for each `model_names` will be made with
        the same `func_args`. By default None.
    func_kwargs : dict | None, optional
        A dict of additional keyword arguments to pass to `target_func`. Calls for each `model_names` will be
        made with the same `func_kwargs`. By default None.
    """

    if len(model_names) != len(model_footprints):
        raise ValueError('The `model_names` and `model_footprints` arguments should have the same length.')
    
    if any([x > num_gpus for x in model_footprints]):
        raise ValueError('One of the function calls needs more gpus than the total number available `num_gpus`.')
    
    func_kwargs = {} if func_kwargs is None else func_kwargs

    # Need to use spawn to create subprocesses in order to set correctly the CUDA_VISIBLE_DEVICES env variable
    ctx = mp.get_context('spawn')

    # Retrieve the function that will be used (the function created by the decorator)
    target_func_gpu_dispatch = globals()[f'{target_func.__name__}_gpu_dispatch']

    model_names = list(model_names)
    model_footprints = list(model_footprints)

    # Sort both lists according to gpu footprint
    sorting = sorted(zip(model_names, model_footprints), key=lambda x: x[1])
    model_names = [x for x, _ in sorting]
    model_footprints = [x for _, x in sorting]

    # Initialize the lists we will maintain
    available_gpus = [i for i in range(num_gpus)]
    processes = []
    associated_gpus = []

    while True:

        no_sleep = False

        # In this case we have enough gpus available to launch the job that needs the less gpus
        if len(available_gpus) >= model_footprints[0]:

            no_sleep = True

            # Remove them from the list of models to process
            name = model_names.pop(0)
            footprint = model_footprints.pop(0)

            # Update gpu resources
            allocated_gpus = available_gpus[0:footprint]
            available_gpus = available_gpus[footprint:]

            args = (allocated_gpus, name) if func_args is None else (allocated_gpus, name, *func_args)
            p = ctx.Process(target=target_func_gpu_dispatch, args=args, kwargs=func_kwargs)
            p.start()

            # Add them to the list of running processes
            processes.append(p)
            associated_gpus.append(allocated_gpus)

        # Find the indices of the processes that are finished
        indices_to_remove = []
        for i, process in enumerate(processes):
            if not process.is_alive():
                indices_to_remove.append(i)
                process.close()

        # Update gpu resources
        released_gpus = [gpus for i, gpus in enumerate(associated_gpus) if i in indices_to_remove]
        available_gpus += [gpu for gpus in released_gpus for gpu in gpus]
        # Remove processes which are done
        processes = [process for i, process in enumerate(processes) if i not in indices_to_remove]
        associated_gpus = [gpus for i, gpus in enumerate(associated_gpus) if i not in indices_to_remove]

        # If we scheduled all jobs break from the infinite loop
        if len(model_names) == 0:
            break

        # Sleep for 10 seconds before restarting the loop and check if we have enough resources to launch
        # a new job
        if not no_sleep:
            time.sleep(10)

    # Sleep until all processes are finished (they have all been scheduled at this point)
    for process in processes:
        process.join()
