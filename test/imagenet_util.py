# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.layers import utils
import cnn_util

def get_image_resize_method(resize_method, batch_position=0):
    """Get tensorflow resize method.

  If resize_method is 'round_robin', return different methods based on batch
  position in a round-robin fashion. NOTE: If the batch size is not a multiple
  of the number of methods, then the distribution of methods will not be
  uniform.

  Args:
    resize_method: (string) nearest, bilinear, bicubic, area, or round_robin.
    batch_position: position of the image in a batch. NOTE: this argument can
      be an integer or a tensor
  Returns:
    one of resize type defined in tf.image.ResizeMethod.
  """
    resize_methods_map = {
        'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'bicubic': tf.image.ResizeMethod.BICUBIC,
        'area': tf.image.ResizeMethod.AREA
    }

    if resize_method != 'round_robin':
        return resize_methods_map[resize_method]

    # return a resize method based on batch position in a round-robin fashion.
    resize_methods = resize_methods_map.values()

    def lookup(index):
        return resize_methods[index]

    def resize_method_0():
        return utils.smart_cond(batch_position % len(resize_methods) == 0,
                                lambda: lookup(0), resize_method_1)

    def resize_method_1():
        return utils.smart_cond(batch_position % len(resize_methods) == 1,
                                lambda: lookup(1), resize_method_2)

    def resize_method_2():
        return utils.smart_cond(batch_position % len(resize_methods) == 2,
                                lambda: lookup(2), lambda: lookup(3))

    # NOTE(jsimsa): Unfortunately, we cannot use a single recursive function
    # here because TF would not be able to construct a finite graph.

    return resize_method_0()


def parse_example_proto(example_serialized):
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.io.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']


def distort_color(image, batch_position=0, distort_color_in_yiq=False,
                  scope=None):
    with tf.name_scope(scope or 'distort_color'):

        def distort_fn_0(image=image):
            """Variant 0 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            if distort_color_in_yiq:
                image = tfa.image.random_hsv_in_yiq(
                    image, lower_saturation=0.5, upper_saturation=1.5,
                    max_delta_hue=0.2 * math.pi)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            return image

        def distort_fn_1(image=image):
            """Variant 1 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            if distort_color_in_yiq:
                image = tfa.image.random_hsv_in_yiq(
                    image, lower_saturation=0.5, upper_saturation=1.5,
                    max_delta_hue=0.2 * math.pi)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            return image

        image = utils.smart_cond(batch_position % 2 == 0, distort_fn_0,
                                 distort_fn_1)
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def train_image(image_buffer,
                height,
                width,
                bbox,
                batch_position,
                resize_method,
                distortions,
                scope=None,
                summary_verbosity=0,
                distort_color_in_yiq=False,
                fuse_decode_and_crop=False):
    with tf.name_scope(scope or 'distort_image'):
        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of
        # interest.  We choose to create a new bounding box for the object which
        # is a randomly distorted version of the human-annotated bounding box
        # that obeys an allowed range of aspect ratios, sizes and overlap with
        # the human-annotated bounding box. If no box is supplied, then we
        # assume the bounding box is the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_buffer),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        if summary_verbosity >= 3:
            image = tf.image.decode_jpeg(image_buffer, channels=3,
                                         dct_method='INTEGER_FAST')
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image_with_distorted_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image(
                'images_with_distorted_bounding_box',
                image_with_distorted_box)

        # Crop the image to the specified bounding box.
        if fuse_decode_and_crop:
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            crop_window = tf.stack(
                [offset_y, offset_x, target_height, target_width])
            image = tf.image.decode_and_crop_jpeg(
                image_buffer, crop_window, channels=3)
        else:
            image = tf.image.decode_jpeg(image_buffer, channels=3,
                                         dct_method='INTEGER_FAST')
            image = tf.slice(image, bbox_begin, bbox_size)

        distorted_image = tf.image.random_flip_left_right(image)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected.
        image_resize_method = get_image_resize_method(resize_method,
                                                      batch_position)
        if cnn_util.tensorflow_version() >= 11:
            distorted_image = tf.compat.v1.image.resize(
                distorted_image, [height, width],
                image_resize_method,
                align_corners=False)
        else:
            distorted_image = tf.compat.v1.image.resize(
                distorted_image,
                height,
                width,
                image_resize_method,
                align_corners=False)
        # Restore the shape since the dynamic slice based upon the bbox_size
        # loses the third dimension.
        distorted_image.set_shape([height, width, 3])
        if summary_verbosity >= 3:
            tf.summary.image('cropped_resized_maybe_flipped_image',
                             tf.expand_dims(distorted_image, 0))

        if distortions:
            distorted_image = tf.cast(distorted_image, dtype=tf.float32)
            # Images values are expected to be in [0,1] for color distortion.
            distorted_image /= 255.
            # Randomly distort the colors.
            distorted_image = distort_color(
                                distorted_image, batch_position,
                                distort_color_in_yiq=distort_color_in_yiq)

            # Note: This ensures the scaling matches the output of eval_image
            distorted_image *= 255

        if summary_verbosity >= 3:
            tf.summary.image(
                'final_distorted_image',
                tf.expand_dims(distorted_image, 0))
        return distorted_image


def preprocess(image_buffer, bbox, batch_position):
    """Preprocessing image_buffer as a function of its batch position."""
    image = train_image(image_buffer, 224, 224, bbox,
                        batch_position, 'bilinear',
                        True,
                        None, summary_verbosity=0,
                        distort_color_in_yiq=True,
                        fuse_decode_and_crop=True)

    return image


def parse_and_preprocess(value, batch_position):
    image_buffer, label_index, bbox, _ = parse_example_proto(value)
    image = preprocess(image_buffer, bbox, batch_position)
    return (label_index, image)
