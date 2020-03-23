import tensorflow as tf
import numpy as np
import pytest
from functools import reduce

from test_util import *

@initialize_ckpt
def test_imagenet():
    g = tf.Graph()
    with g.as_default():
        n = build_imagenet_input_pipeline(1, 1, 0)

    expected_result = run_steps(g, n, 5004)
    do_initialize_ckpt()
    res = run_steps(g, n, 1000)
    res += run_steps(g, n, 4004)
    assert all([e == r for e, r in zip(expected_result, res)])

@initialize_ckpt
def test_imagenet_2():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 2, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(1, 2, 1, repeat=False)

    expected_result = reduce(lambda x,y:x+y,
                             run_steps(g, [n1, n2], 2502)) # one epoch
    do_initialize_ckpt()
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 4, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(1, 4, 1, repeat=False)
        n3 = build_imagenet_input_pipeline(1, 4, 2, repeat=False)
        n4 = build_imagenet_input_pipeline(1, 4, 3, repeat=False)
    res = reduce(lambda x,y:x+y, run_steps(g, [n1, n2, n3, n4], 1251))
    assert set_eq(expected_result, res)

@initialize_ckpt
def test_imagenet_3():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 2, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(1, 2, 1, repeat=False)

    expected_result = reduce(lambda x,y:x+y,
                             run_steps(g, [n1, n2], 2502)) # one epoch
    do_initialize_ckpt()
    res = reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 1000))
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 1502))
    assert all([e == r for e, r in zip(expected_result, res)])

@initialize_ckpt
def test_imagenet_scale_out():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 2, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(1, 2, 1, repeat=False)

    expected_result = reduce(lambda x,y:x+y,
                             run_steps(g, [n1, n2], 2502)) # one epoch
    do_initialize_ckpt()
    res = reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 100))

    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 4, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(1, 4, 1, repeat=False)
        n3 = build_imagenet_input_pipeline(1, 4, 2, repeat=False)
        n4 = build_imagenet_input_pipeline(1, 4, 3, repeat=False)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1], 1151))
    res += reduce(lambda x,y:x+y, run_steps(g, [n2], 1151))
    res += reduce(lambda x,y:x+y, run_steps(g, [n3], 1251))
    res += reduce(lambda x,y:x+y, run_steps(g, [n4], 1251))
    assert set_eq(expected_result, res)

@initialize_ckpt
def test_imagenet_scale_in():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 4, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(1, 4, 1, repeat=False)
        n3 = build_imagenet_input_pipeline(1, 4, 2, repeat=False)
        n4 = build_imagenet_input_pipeline(1, 4, 3, repeat=False)
    expected_result = reduce(lambda x,y:x+y,
                             run_steps(g, [n1, n2, n3, n4], 1251)) # one epoch
    do_initialize_ckpt()
    res = reduce(lambda x,y:x+y, run_steps(g, [n1,n2,n3,n4], 625))

    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 2, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(1, 2, 1, repeat=False)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 1252))
    assert set_eq(expected_result, res)
