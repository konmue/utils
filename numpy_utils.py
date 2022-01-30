import numpy as np


def np_stack(arr: np.ndarray, repeat: int, dim=0):
    for i in range(repeat):
        if i == 0:
            stack = np.expand_dims(arr, dim)
        else:
            if dim == 0:
                stack = np.r_[stack, np.expand_dims(arr, dim)]
            elif dim == 1:
                stack = np.c_[stack, np.expand_dims(arr, dim)]
    return stack
