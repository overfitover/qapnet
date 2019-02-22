# import tensorflow as tf
# from keras import backend as K
# import numpy as np
#
# boxes = np.array([[1,2,3,4], [1,3,4,4], [1,1,4,4], [1,1,3,4]], dtype=np.float32)
# scores = np.array([0.4, 0.5, 0.72, 0.9, 0.45], dtype=np.float32)
# # scores = [0.4]
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     selected_indices=sess.run(tf.image.non_max_suppression(boxes=boxes, scores=scores, iou_threshold=0.5, max_output_size=5))
#     print(selected_indices)
#     selected_boxes = sess.run(tf.gather(boxes, selected_indices))
#     print(selected_boxes)


import tensorflow as tf
import numpy as np

rects=np.asarray([[1,2,3,4],[1,3,3,4], [1,3,4,4],[1,1,4,4],[1,1,3,4]],dtype=np.float32)
scores=np.asarray([0.4,0.5,0.72,0.9,0.45], dtype=np.float32)

with tf.Session() as sess:
    nms = tf.image.non_max_suppression(rects, scores, max_output_size=5, iou_threshold=0.5)
    print(sess.run(nms))
    selected_boxes = sess.run(tf.gather(rects, nms))
    print(selected_boxes)


