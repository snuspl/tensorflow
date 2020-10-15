import tensorflow as tf
import numpy as np
import pytest

import test_util

@test_util.initialize_ckpt
def test_imagenet_1():
    g = tf.Graph()
    with g.as_default():
        iterator = test_util.build_imagenet_input_pipeline(1, 1, 0)

    expected_result = test_util.flatten(test_util.run_steps(g, [iterator], 5004))
    test_util.do_initialize_ckpt()
    res = test_util.flatten(test_util.run_steps(g, [iterator], 1000))
    res += test_util.flatten(test_util.run_steps(g, [iterator], 4004))
    assert test_util.set_eq(expected_result, res)

@test_util.initialize_ckpt
def test_imagenet_2():
    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(2, 2, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(2, 2, 1, repeat=False)

    expected_result = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2], 1251)) # one epoch
    test_util.do_initialize_ckpt()
    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(1, 4, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(1, 4, 1, repeat=False)
        iterator_3 = test_util.build_imagenet_input_pipeline(1, 4, 2, repeat=False)
        iterator_4 = test_util.build_imagenet_input_pipeline(1, 4, 3, repeat=False)
    res = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2, iterator_3, iterator_4], 1251))
    assert test_util.set_eq(expected_result, res)

@test_util.initialize_ckpt
def test_imagenet_3():
    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(1, 2, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(1, 2, 1, repeat=False)

    expected_result = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2], 2502)) # one epoch
    test_util.do_initialize_ckpt()
    res = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2], 1000))
    res += test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2], 1502))
    assert test_util.set_eq(expected_result, res)

@test_util.initialize_ckpt
def test_imagenet_scale_out_1():
    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(6, 2, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(6, 2, 1, repeat=False)

    expected_result = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2], 417)) # one epoch
    test_util.do_initialize_ckpt()
    res = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2], 100))

    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(3, 4, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(3, 4, 1, repeat=False)
        iterator_3 = test_util.build_imagenet_input_pipeline(3, 4, 2, repeat=False)
        iterator_4 = test_util.build_imagenet_input_pipeline(3, 4, 3, repeat=False)
    res += test_util.flatten(test_util.run_steps(g, [iterator_1], 217))
    res += test_util.flatten(test_util.run_steps(g, [iterator_2], 217))
    res += test_util.flatten(test_util.run_steps(g, [iterator_3], 417))
    res += test_util.flatten(test_util.run_steps(g, [iterator_4], 417))
    assert test_util.set_eq(expected_result, res)

@test_util.initialize_ckpt
def test_imagenet_scale_out_2():
    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(6, 2, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(6, 2, 1, repeat=False)

    expected_result = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2], 417)) # one epoch
    test_util.do_initialize_ckpt()
    res = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2], 100))

    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(4, 3, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(4, 3, 1, repeat=False)
        iterator_3 = test_util.build_imagenet_input_pipeline(4, 3, 2, repeat=False)
    res += test_util.flatten(test_util.run_steps(g, [iterator_1], 476))
    res += test_util.flatten(test_util.run_steps(g, [iterator_2], 163))
    res += test_util.flatten(test_util.run_steps(g, [iterator_3], 313))
    assert test_util.set_eq(expected_result, res)

@test_util.initialize_ckpt
def test_imagenet_scale_in_1():
    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(3, 4, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(3, 4, 1, repeat=False)
        iterator_3 = test_util.build_imagenet_input_pipeline(3, 4, 2, repeat=False)
        iterator_4 = test_util.build_imagenet_input_pipeline(3, 4, 3, repeat=False)
    expected_result = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2, iterator_3, iterator_4], 417)) # one epoch
    test_util.do_initialize_ckpt()
    res = test_util.flatten(test_util.run_steps(g, [iterator_1,iterator_2,iterator_3,iterator_4], 100))

    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(6, 2, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(6, 2, 1, repeat=False)
    res += test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2], 317))
    assert test_util.set_eq(expected_result, res)

@test_util.initialize_ckpt
def test_imagenet_scale_in_2():
    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(3, 4, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(3, 4, 1, repeat=False)
        iterator_3 = test_util.build_imagenet_input_pipeline(3, 4, 2, repeat=False)
        iterator_4 = test_util.build_imagenet_input_pipeline(3, 4, 3, repeat=False)
    expected_result = test_util.flatten(test_util.run_steps(g, [iterator_1, iterator_2, iterator_3, iterator_4], 417)) # one epoch
    test_util.do_initialize_ckpt()
    res = test_util.flatten(test_util.run_steps(g, [iterator_1,iterator_2,iterator_3,iterator_4], 100))

    g = tf.Graph()
    with g.as_default():
        iterator_1 = test_util.build_imagenet_input_pipeline(4, 3, 0, repeat=False)
        iterator_2 = test_util.build_imagenet_input_pipeline(4, 3, 1, repeat=False)
        iterator_3 = test_util.build_imagenet_input_pipeline(4, 3, 2, repeat=False)
    res += test_util.flatten(test_util.run_steps(g, [iterator_1], 476))
    res += test_util.flatten(test_util.run_steps(g, [iterator_2], 238))
    res += test_util.flatten(test_util.run_steps(g, [iterator_3], 238))
    assert test_util.set_eq(expected_result, res)
