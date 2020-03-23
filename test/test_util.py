import tensorflow as tf
import os
import glob
import time

def get_ckpt_dir():
    return "/tmp/eparallax-{}/checkpoint/index/".format(os.environ["USER"])

def do_initialize_ckpt():
    for f in glob.glob(get_ckpt_dir() + "*"):
        os.remove(f)

def initialize_ckpt(fn):
    def wrapper(*args, **kwargs):
        do_initialize_ckpt()
        return fn(*args, **kwargs)
    return wrapper

def aggregate_ckpt(fn):
    def wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)
        with open(get_ckpt_dir() + "index_ckpt", 'a') as global_ckpt:
            for f in glob.glob(get_ckpt_dir() + "index_ckpt_*"):
                with open(f, 'r') as shard_ckpt:
                    global_ckpt.write(shard_ckpt.read())
                os.remove(f)
        return ret
    return wrapper

@aggregate_ckpt
def run_steps(graph, n, num_steps, initializer=None, measure_time=True):
    start = None
    if measure_time:
        start = time.time()
    with tf.compat.v1.Session(graph=graph) as sess:
        if initializer is not None:
            sess.run(initializer)
        res = []
        for i in range(num_steps):
            res.append(sess.run(n))
            #print(i)
            #print(res[-1])
        if measure_time:
            print(time.time() - start)
        return res

def build_imagenet_input_pipeline(batch_size, num_workers, worker_id,
                                  repeat=True, shuffle=False):
    file_names = [
        "/cmsdata/ssd1/cmslab/imagenet-data/aws/train-00{}-of-01024"
            .format(i) for i in range(256, 256+4)
    ]
    file_names.sort()
    num_splits = 1
    batch_size_per_split = batch_size // num_splits
    ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=shuffle, seed=1)
    #ds = tf.data.Dataset.from_tensor_slices(file_names)
    ds_ = ds.shard(num_workers, worker_id)
    ds = ds_.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            #tf.data.Dataset.from_tensors,
            cycle_length=1))
    counter = tf.data.Dataset.range(batch_size)
    counter = counter.repeat()
    ds = tf.data.Dataset.zip((ds, counter))
    ds = ds.prefetch(buffer_size=batch_size)
    if (shuffle):
        ds = ds.shuffle(buffer_size=50)
    if (repeat):
        ds = ds.repeat()
    #ds = ds.apply(
    #    tf.data.experimental.map_and_batch(
    #        map_func=lambda *x:x,
    #        batch_size=batch_size_per_split,
    #        num_parallel_batches=num_splits))
    ds = ds.prefetch(buffer_size=num_splits)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    return iterator.get_next()

def set_eq(list_1, list_2):
    #for e1, e2 in zip(list_1, list_2):
    #    print(e1[0], e2[0])

    #if len(list_1) != len(list_2):
    #    return False

    for i, elem_1 in enumerate(list_1):
        found = False
        for j, elem_2 in enumerate(list_2):
            if elem_1[0] == elem_2[0]:
                print("list_1[{}] == list_2[{}]".format(i, j))
                #print("{} == {}".format(elem_1[2], elem_2[2]))
                found = True
                break
        if not found:
            print("Not found: list_1[{}]".format(i))
            #return False

    for i, elem_2 in enumerate(list_2):
        found = False
        for j, elem_1 in enumerate(list_1):
            if elem_1[0] == elem_2[0]:
                print("list_2[{}] == list_1[{}]".format(i, j))
                #print("{} == {}".format(elem_1[2], elem_2[2]))
                found = True
                break
        if not found:
            print("Not found: list_2[{}]".format(i))
            #print("Not found: {}".format(elem_2[2]))
            #return False

    return True
