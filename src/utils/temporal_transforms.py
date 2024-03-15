import random
import math
import numpy as np


class LoopPadding(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, frame_indices):
        out = frame_indices
        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)
        return out

def duplicate_random_index(size,value): //tang so luong frame de dat duoc tong so frame mong muon (size = so frame muon tang len mong muon, value = tong so frame)
    index = sorted(np.random.randint(0,value,size).tolist()) //gia su: frame sau khi random ra bi xao tron (vd 0 2 1 1 2 0) => sort lại để sắp xếp để hành động diễn ra đúng trình tự
    return index
class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.
    
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return np.float32(out)


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        if len(frame_indices)< self.size:
            index = duplicate_random_index(self.size)
            out = index
        else:
            center_index = len(frame_indices) // 2
            begin_index = max(0, center_index - (self.size // 2))
            end_index = min(begin_index + self.size, len(frame_indices))
            out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return np.float32(out)


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        if len(frames)<self.size:
            out_index = duplicate_random_index(self.size)
        else:
            frame_indices = list(range(frames.shape[0]))
            rand_end = max(0, frames.shape[0] - self.size - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.size, len(frame_indices))
            out_index = frame_indices[begin_index:end_index]
        out = []

        for index in out_index:
            if len(out_index) >= self.size:
                break
            out_index.append(index)

        for index in out_index:
            out.append(frames[index])
        return np.float32(out)
