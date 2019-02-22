# import numpy as np
# import easydict
# import tensorflow as tf
#
# # dictory = {'Google': 1, 'Runoob': 2, 'taobao': 3}
# # a = dict()
# # for key, value in dictory.items():
# #     a[value] = dictory[key]
# #     print(a)
#
# def get_box_indices(boxes):
#     proposals_shape = boxes.get_shape().as_list()                 # (5, 7, 3)
#     if any(dim is None for dim in proposals_shape):
#         proposals_shape = tf.shape(boxes)
#     ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)       # (5, 7)
#     multiplier = tf.expand_dims(
#         tf.range(start=0, limit=proposals_shape[0]), 1)           # (5, 1)
#     a = (ones_mat * multiplier)                                   # (5, 7)
#     return tf.reshape(ones_mat * multiplier, [-1])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     boxes = tf.ones([5, 7, 3])
#     b = get_box_indices(boxes)   # 35
#     print(b)

import os
is_travis = 'TRAVIS' in os.environ
print(is_travis)