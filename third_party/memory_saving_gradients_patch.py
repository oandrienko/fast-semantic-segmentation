r"""Monkey patch for using memory saving gradients package from OpenAI"""
from tensorflow.python.ops import gradients
from third_party import memory_saving_gradients

CHECKPOINT_TYPES = "collection"

TF_GRADIENTS_KEY = "gradients"

def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(
        ys, xs, grad_ys, checkpoints=CHECKPOINT_TYPES, **kwargs)
gradients.__dict__[TF_GRADIENTS_KEY] = gradients_memory
