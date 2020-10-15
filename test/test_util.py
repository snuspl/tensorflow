import tensorflow as tf
import os
import glob
import time
from functools import reduce
import imagenet_util


CHECKPOINT_PATH = '/tmp/tf-elastic-input-pipeline-test/idx_ckpt'
if not os.path.exists(os.path.dirname(CHECKPOINT_PATH)):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH))

def do_initialize_ckpt():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

def initialize_ckpt(fn):
    def wrapper(*args, **kwargs):
        do_initialize_ckpt()
        return fn(*args, **kwargs)
    return wrapper

def aggregate_ckpt(local_ckpt_paths, global_ckpt_path):
    with open(global_ckpt_path, 'w') as global_ckpt:
        for ckpt_path in local_ckpt_paths:
            with open(ckpt_path, 'r') as local_ckpt:
                global_ckpt.write(local_ckpt.read())
            os.remove(ckpt_path)

def flatten(res):
    return reduce(lambda x,y:x+y,
            [r[0].tolist() for r in reduce(lambda x,y:x+y, res)])

def run_steps(graph, iterators, num_steps, initializer=None, measure_time=True):
    start = None

    flatten = False
    if not isinstance(iterators, list):
        iterators = [iterators]
        flatten = True

    global_ckpt_path = CHECKPOINT_PATH
    local_ckpt_paths = [CHECKPOINT_PATH + f'_{i}' for i, _ in enumerate(iterators)]

    with graph.as_default():
        get_next_ops = [iterator.get_next() for iterator in iterators]
        restore_checkpoint_ops = [
            iterator.restore_checkpoint(global_ckpt_path)
            for iterator in iterators
        ]
        save_checkpoint_ops = [
            iterator.save_checkpoint(ckpt_path)
            for iterator, ckpt_path in zip(iterators, local_ckpt_paths)
        ]

    if measure_time:
        start = time.time()
    with tf.compat.v1.Session(graph=graph) as sess:
        if initializer is not None:
            sess.run(initializer)
        sess.run(restore_checkpoint_ops)
        res = []
        for i in range(num_steps):
            res.append(sess.run(get_next_ops))
        if measure_time:
            print(time.time() - start)
        sess.run(save_checkpoint_ops)

        aggregate_ckpt(local_ckpt_paths, global_ckpt_path)

        if flatten:
            res = [r[0] for r in res]
        return res

def build_imagenet_input_pipeline(batch_size, num_workers, worker_id,
                                  repeat=True, shuffle=True, preprocess=False):
    file_names = [
        "/cmsdata/ssd1/cmslab/imagenet-data/train-{:05d}-of-01024".format(i)
        for i in range(0, 4)
    ]
    file_names.sort()
    num_splits = 1
    batch_size_per_split = batch_size // num_splits
    ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)
    ds_ = ds.shard(num_workers, worker_id)
    ds = ds_.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=1))
    counter = tf.data.Dataset.range(batch_size)
    counter = counter.repeat()
    ds = tf.data.Dataset.zip((ds, counter))
    ds = ds.prefetch(buffer_size=batch_size)
    if (shuffle):
        ds = ds.shuffle(buffer_size=50)
    if (repeat):
        ds = ds.repeat()
    map_fn = imagenet_util.parse_and_preprocess if preprocess else lambda *x: x
    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            map_func=map_fn,
            batch_size=batch_size_per_split,
            num_parallel_batches=num_splits))
    ds = ds.prefetch(buffer_size=num_splits)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    return iterator

def set_eq(list_1, list_2):
    eq = True

    if len(list_1) != len(list_2):
        eq = False

    for i, elem_1 in enumerate(list_1):
        found = False
        for j, elem_2 in enumerate(list_2):
            if elem_1 == elem_2:
                print("list_1[{}] == list_2[{}]".format(i, j))
                found = True
                break
        if not found:
            print("Not found: list_1[{}]".format(i))
            eq = False

    for i, elem_2 in enumerate(list_2):
        found = False
        for j, elem_1 in enumerate(list_1):
            if elem_1 == elem_2:
                print("list_2[{}] == list_1[{}]".format(i, j))
                found = True
                break
        if not found:
            print("Not found: list_2[{}]".format(i))
            eq = False

    return eq
