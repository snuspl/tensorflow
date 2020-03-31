import tensorflow as tf
import numpy as np
import pytest

from test_util import *

@initialize_ckpt
def test_imagenet_1():
    g = tf.Graph()
    with g.as_default():
        n = build_imagenet_input_pipeline(1, 1, 0)

    expected_result = flatten(run_steps(g, [n], 5004))
    do_initialize_ckpt()
    res = flatten(run_steps(g, [n], 1000))
    res += flatten(run_steps(g, [n], 4004))
    assert set_eq(expected_result, res)

@initialize_ckpt
def test_imagenet_2():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(2, 2, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(2, 2, 1, repeat=False)

    expected_result = flatten(run_steps(g, [n1, n2], 1251)) # one epoch
    do_initialize_ckpt()
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 4, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(1, 4, 1, repeat=False)
        n3 = build_imagenet_input_pipeline(1, 4, 2, repeat=False)
        n4 = build_imagenet_input_pipeline(1, 4, 3, repeat=False)
    res = flatten(run_steps(g, [n1, n2, n3, n4], 1251))
    assert set_eq(expected_result, res)

@initialize_ckpt
def test_imagenet_3():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 2, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(1, 2, 1, repeat=False)

    expected_result = flatten(run_steps(g, [n1, n2], 2502)) # one epoch
    do_initialize_ckpt()
    res = flatten(run_steps(g, [n1, n2], 1000))
    res += flatten(run_steps(g, [n1, n2], 1502))
    assert set_eq(expected_result, res)

@initialize_ckpt
def test_imagenet_scale_out_1():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(6, 2, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(6, 2, 1, repeat=False)

    expected_result = flatten(run_steps(g, [n1, n2], 417)) # one epoch
    do_initialize_ckpt()
    res = flatten(run_steps(g, [n1, n2], 100))

    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(3, 4, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(3, 4, 1, repeat=False)
        n3 = build_imagenet_input_pipeline(3, 4, 2, repeat=False)
        n4 = build_imagenet_input_pipeline(3, 4, 3, repeat=False)
    res += flatten(run_steps(g, [n1], 217))
    res += flatten(run_steps(g, [n2], 217))
    res += flatten(run_steps(g, [n3], 417))
    res += flatten(run_steps(g, [n4], 417))
    assert set_eq(expected_result, res)

@initialize_ckpt
def test_imagenet_scale_out_2():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(6, 2, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(6, 2, 1, repeat=False)

    expected_result = flatten(run_steps(g, [n1, n2], 417)) # one epoch
    do_initialize_ckpt()
    res = flatten(run_steps(g, [n1, n2], 100))

    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(4, 3, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(4, 3, 1, repeat=False)
        n3 = build_imagenet_input_pipeline(4, 3, 2, repeat=False)
    res += flatten(run_steps(g, [n1], 476))
    res += flatten(run_steps(g, [n2], 163))
    res += flatten(run_steps(g, [n3], 313))
    assert set_eq(expected_result, res)

@initialize_ckpt
def test_imagenet_scale_in_1():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(3, 4, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(3, 4, 1, repeat=False)
        n3 = build_imagenet_input_pipeline(3, 4, 2, repeat=False)
        n4 = build_imagenet_input_pipeline(3, 4, 3, repeat=False)
    expected_result = flatten(run_steps(g, [n1, n2, n3, n4], 417)) # one epoch
    do_initialize_ckpt()
    res = flatten(run_steps(g, [n1,n2,n3,n4], 100))

    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(6, 2, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(6, 2, 1, repeat=False)
    res += flatten(run_steps(g, [n1, n2], 317))
    assert set_eq(expected_result, res)

@initialize_ckpt
def test_imagenet_scale_in_2():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(3, 4, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(3, 4, 1, repeat=False)
        n3 = build_imagenet_input_pipeline(3, 4, 2, repeat=False)
        n4 = build_imagenet_input_pipeline(3, 4, 3, repeat=False)
    expected_result = flatten(run_steps(g, [n1, n2, n3, n4], 417)) # one epoch
    do_initialize_ckpt()
    res = flatten(run_steps(g, [n1,n2,n3,n4], 100))

    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(4, 3, 0, repeat=False)
        n2 = build_imagenet_input_pipeline(4, 3, 1, repeat=False)
        n3 = build_imagenet_input_pipeline(4, 3, 2, repeat=False)
    res += flatten(run_steps(g, [n1], 476))
    res += flatten(run_steps(g, [n2], 238))
    res += flatten(run_steps(g, [n3], 238))
    assert set_eq(expected_result, res)
