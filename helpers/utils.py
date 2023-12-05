import os

# Path to the root
ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))

# Path to the few shot examples yaml templates
FEW_SHOT_FOLDER = os.path.join(ROOT_FOLDER, 'few_shot_examples')


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
    