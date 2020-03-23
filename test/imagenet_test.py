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

    expected_result = run_steps(g, n, 100)
    do_initialize_ckpt()
    res = run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    #for e, r in zip(expected_result, res):
    #    print(e == r, e, r)
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
    print(1)
    do_initialize_ckpt()
    res = reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(2)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(3)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(4)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 352))
    print(5)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(6)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(7)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(8)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 350))
    print(9)
    assert all([e == r for e, r in zip(expected_result, res)])

@initialize_ckpt
def test_imagenet_scale_out():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 2, 0, repeat=False, shuffle=True)
        n2 = build_imagenet_input_pipeline(1, 2, 1, repeat=False, shuffle=True)

    expected_result = reduce(lambda x,y:x+y,
                             run_steps(g, [n1, n2], 2502)) # one epoch
    print(1)
    do_initialize_ckpt()
    res = reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(2)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(3)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(4)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 352))
    print(5)

    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 4, 0, repeat=False, shuffle=True)
        n2 = build_imagenet_input_pipeline(1, 4, 1, repeat=False, shuffle=True)
        n3 = build_imagenet_input_pipeline(1, 4, 2, repeat=False, shuffle=True)
        n4 = build_imagenet_input_pipeline(1, 4, 3, repeat=False, shuffle=True)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1,n2,n3,n4], 150))
    print(6)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1,n2,n3,n4], 150))
    print(7)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1,n2,n3,n4], 150))
    print(8)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1,n2,n3,n4], 175))
    print(9)
    assert set_eq(expected_result, res)

@initialize_ckpt
def test_imagenet_scale_in():
    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 4, 0, repeat=False, shuffle=True)
        n2 = build_imagenet_input_pipeline(1, 4, 1, repeat=False, shuffle=True)
        n3 = build_imagenet_input_pipeline(1, 4, 2, repeat=False, shuffle=True)
        n4 = build_imagenet_input_pipeline(1, 4, 3, repeat=False, shuffle=True)
    expected_result = reduce(lambda x,y:x+y,
                             run_steps(g, [n1, n2, n3, n4], 1251)) # one epoch
    print(1)
    do_initialize_ckpt()
    res = reduce(lambda x,y:x+y, run_steps(g, [n1,n2,n3,n4], 150))
    print(2)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1,n2,n3,n4], 150))
    print(3)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1,n2,n3,n4], 150))
    print(4)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1,n2,n3,n4], 175))
    print(5)

    g = tf.Graph()
    with g.as_default():
        n1 = build_imagenet_input_pipeline(1, 2, 0, repeat=False, shuffle=True)
        n2 = build_imagenet_input_pipeline(1, 2, 1, repeat=False, shuffle=True)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(6)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(7)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 300))
    print(8)
    res += reduce(lambda x,y:x+y, run_steps(g, [n1, n2], 352))
    print(9)
    assert set_eq(expected_result, res)

if __name__ == "__main__":
    #import time
    #time.sleep(10)
    #test_imagenet_scale_out()
    #test_imagenet_scale_in()
    #test_imagenet()
    test_imagenet_3()
