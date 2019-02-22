import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import sys
# sys.path.append('/home/yxk/project/aa_demo_graduate/qapNet')
# sys.path.append('/home/yxk/project/aa_demo_graduate/qapNet/wavedata')

from avod.builders import feature_extractor_builder
from avod.core import anchor_encoder
from avod.core import anchor_filter
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core import constants
from avod.core import losses
from avod.core import model
from avod.core import summary_utils
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.datasets.kitti import kitti_aug



import argparse
import avod
import avod.builders.config_builder_util as config_builder
import os
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import trainer

tf.logging.set_verbosity(tf.logging.ERROR)


class RpnModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_BEV_INPUT = 'bev_input_pl'
    # PL_IMG_INPUT = 'img_input_pl'
    PL_ANCHORS = 'anchors_pl'

    PL_BEV_ANCHORS = 'bev_anchors_pl'
    PL_BEV_ANCHORS_NORM = 'bev_anchors_norm_pl'

    PL_IMG_ANCHORS = 'img_anchors_pl'
    PL_IMG_ANCHORS_NORM = 'img_anchors_norm_pl'

    PL_LABEL_ANCHORS = 'label_anchors_pl'
    PL_LABEL_BOXES_3D = 'label_boxes_3d_pl'
    PL_LABEL_CLASSES = 'label_classes_pl'

    PL_ANCHOR_IOUS = 'anchor_ious_pl'
    PL_ANCHOR_OFFSETS = 'anchor_offsets_pl'
    PL_ANCHOR_CLASSES = 'anchor_classes_pl'

    # Sample info, including keys for projection to image space
    # (e.g. camera matrix, image index, etc.)
    PL_CALIB_P2 = 'frame_calib_p2'
    PL_IMG_IDX = 'current_img_idx'
    PL_GROUND_PLANE = 'ground_plane'

    ##############################
    # Keys for Predictions
    ##############################
    PRED_ANCHORS = 'rpn_anchors'

    PRED_MB_OBJECTNESS_GT = 'rpn_mb_objectness_gt'
    PRED_MB_OFFSETS_GT = 'rpn_mb_offsets_gt'

    PRED_MB_MASK = 'rpn_mb_mask'
    PRED_MB_OBJECTNESS = 'rpn_mb_objectness'
    PRED_MB_OFFSETS = 'rpn_mb_offsets'

    PRED_TOP_INDICES = 'rpn_top_indices'
    PRED_TOP_ANCHORS = 'rpn_top_anchors'
    PRED_TOP_OBJECTNESS_SOFTMAX = 'rpn_top_objectness_softmax'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_RPN_OBJECTNESS = 'rpn_objectness_loss'
    LOSS_RPN_REGRESSION = 'rpn_regression_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(RpnModel, self).__init__(model_config)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test                        # 'train'

        self._is_training = (self._train_val_test == 'train')

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])     # (700, 800)
        self._bev_depth = input_config.bev_depth                         # 6

        # self._img_pixel_size = np.asarray([input_config.img_dims_h,
        #                                    input_config.img_dims_w])     # 480, 1590
        # self._img_depth = input_config.img_depth                         # 3


        # Rpn config
        rpn_config = self._config.rpn_config
        self._proposal_roi_crop_size = \
            [rpn_config.rpn_proposal_roi_crop_size] * 2       # 3  [3, 3]
        self._fusion_method = rpn_config.rpn_fusion_method    # 'mean'

        if self._train_val_test in ["train", "val"]:
            self._nms_size = rpn_config.rpn_train_nms_size    # 1024
        else:
            self._nms_size = rpn_config.rpn_test_nms_size     # 300

        self._nms_iou_thresh = rpn_config.rpn_nms_iou_thresh  # 0.8

        # Feature Extractor Nets
        self._bev_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.bev_feature_extractor)   # (350, 400, 256)
        # self._img_feature_extractor = \
        #     feature_extractor_builder.get_extractor(
        #         self._config.layers_config.img_feature_extractor)   # (240, 795, 256)


        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self.sample_info = dict()

        # Dataset
        self.dataset = dataset
        self.dataset.train_val_test = self._train_val_test              # 'train'
        self._area_extents = self.dataset.kitti_utils.area_extents      # [[-40, 40],[-5, 3], [0, 70]]
        self._bev_extents = self.dataset.kitti_utils.bev_extents        # [[-40, 40], [0, 70]]
        self._cluster_sizes, _ = self.dataset.get_cluster_info()        # <class 'list'>: [[array([3.513, 1.581, 1.511]), array([4.234, 1.653, 1.546])]]
        self._anchor_strides = self.dataset.kitti_utils.anchor_strides  # [[0.5, 0.5]]
        self._anchor_generator = \
            grid_anchor_3d_generator.GridAnchor3dGenerator()

        # self._path_drop_probabilities = self._config.path_drop_probabilities  # [0.9, 0.9]
        self._train_on_all_samples = self._config.train_on_all_samples        # false
        self._eval_all_samples = self._config.eval_all_samples                # false
        # Overwrite the dataset's variable with the config
        self.dataset.train_on_all_samples = self._train_on_all_samples

        # if self._train_val_test in ["val", "test"]:   # 在训练为0.9  测试时为1
        #     # Disable path-drop, this should already be disabled inside the
        #     # evaluator, but just in case.
        #     self._path_drop_probabilities[0] = 1.0
        #     self._path_drop_probabilities[1] = 1.0

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        # Combine config data
        bev_dims = np.append(self._bev_pixel_size, self._bev_depth)   # (700, 800, 6)

        with tf.variable_scope('bev_input'):
            # Placeholder for BEV image input, to be filled in with feed_dict
            bev_input_placeholder = self._add_placeholder(tf.float32, bev_dims,
                                                          self.PL_BEV_INPUT)

            self._bev_input_batches = tf.expand_dims(
                bev_input_placeholder, axis=0)                      # 增加一个维度

            self._bev_preprocessed = \
                self._bev_feature_extractor.preprocess_input(
                    self._bev_input_batches, self._bev_pixel_size)  # 输入尺寸不匹配会reshape一下图片尺寸

            # Summary Images
            bev_summary_images = tf.split(
                bev_input_placeholder, self._bev_depth, axis=2)
            tf.summary.image("bev_maps", bev_summary_images,
                             max_outputs=self._bev_depth)

        # with tf.variable_scope('img_input'):
        #     # Take variable size input images
        #     img_input_placeholder = self._add_placeholder(
        #         tf.float32,
        #         [None, None, self._img_depth],
        #         self.PL_IMG_INPUT)
        #
        #     self._img_input_batches = tf.expand_dims(
        #         img_input_placeholder, axis=0)
        #
        #     self._img_preprocessed = \
        #         self._img_feature_extractor.preprocess_input(
        #             self._img_input_batches, self._img_pixel_size)
        #
        #     # Summary Image
        #     tf.summary.image("rgb_image", self._img_preprocessed,
        #                      max_outputs=2)

        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_LABEL_ANCHORS)
            self._add_placeholder(tf.float32, [None, 7],
                                  self.PL_LABEL_BOXES_3D)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_LABEL_CLASSES)

        # Placeholders for anchors
        with tf.variable_scope('pl_anchors'):
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_ANCHORS)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_ANCHOR_IOUS)
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_ANCHOR_OFFSETS)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_ANCHOR_CLASSES)

            with tf.variable_scope('bev_anchor_projections'):
                self._add_placeholder(tf.float32, [None, 4],
                                      self.PL_BEV_ANCHORS)
                self._bev_anchors_norm_pl = self._add_placeholder(
                    tf.float32, [None, 4], self.PL_BEV_ANCHORS_NORM)

            # with tf.variable_scope('img_anchor_projections'):
            #     self._add_placeholder(tf.float32, [None, 4],
            #                           self.PL_IMG_ANCHORS)
            #     self._img_anchors_norm_pl = self._add_placeholder(
            #         tf.float32, [None, 4], self.PL_IMG_ANCHORS_NORM)

            with tf.variable_scope('sample_info'):
                # the calib matrix shape is (3 x 4)
                self._add_placeholder(
                    tf.float32, [3, 4], self.PL_CALIB_P2)
                # self._add_placeholder(tf.int32,
                #                       shape=[1],
                #                       name=self.PL_IMG_IDX)
                self._add_placeholder(tf.float32, [4], self.PL_GROUND_PLANE)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and
        bottlenecks as member variables.
        """

        self.bev_feature_maps, self.bev_end_points = \
            self._bev_feature_extractor.build(
                self._bev_preprocessed,
                self._bev_pixel_size,
                self._is_training)                          # (350,400,256)

        # print("fjsgd", self.bev_feature_maps, self.bev_end_points)

        # self.img_feature_maps, self.img_end_points = \
        #     self._img_feature_extractor.build(
        #         self._img_preprocessed,
        #         self._img_pixel_size,
        #         self._is_training)                          # (240,795,256)

        with tf.variable_scope('bev_bottleneck'):     # bev_bottleneck/bottleneck/Relu:0, shape=(1, 350, 400, 1)
            self.bev_bottleneck = slim.conv2d(
                self.bev_feature_maps,
                1, [1, 1],
                scope='bottleneck',
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'is_training': self._is_training})

        # with tf.variable_scope('img_bottleneck'):      # img_bottleneck/bottleneck/Relu:0, shape=(1, 240, 795, 1)
        #     self.img_bottleneck = slim.conv2d(
        #         self.img_feature_maps,
        #         1, [1, 1],
        #         scope='bottleneck',
        #         normalizer_fn=slim.batch_norm,
        #         normalizer_params={
        #             'is_training': self._is_training})


        # print(self.bev_bottleneck, '\n', self.img_bottleneck)

        # # Visualize the end point feature maps being used
        # for feature_map in list(self.bev_end_points.items()):
        #     if 'conv' in feature_map[0]:
        #         summary_utils.add_feature_maps_from_dict(self.bev_end_points,
        #                                                  feature_map[0])
        #
        # for feature_map in list(self.img_end_points.items()):
        #     if 'conv' in feature_map[0]:
        #         summary_utils.add_feature_maps_from_dict(self.img_end_points,
        #                                                  feature_map[0])

    def build(self):

        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()

        bev_proposal_input = self.bev_bottleneck   # shape=(1, 350, 400, 1)   1*1卷积得到的结果 特征提取网络的输出
        # img_proposal_input = self.img_bottleneck   # shape=(1, 240, 795, 1)

        # fusion_mean_div_factor = 1.0

        # If both img and bev probabilites are set to 1.0, don't do
        # path drop.
        # if not (self._path_drop_probabilities[0] ==
        #         self._path_drop_probabilities[1] == 1.0):
        #     with tf.variable_scope('rpn_path_drop'):
        #
        #         random_values = tf.random_uniform(shape=[3],
        #                                           minval=0.0,
        #                                           maxval=1.0)  # 随机产生3个值
        #
        #         img_mask, bev_mask = self.create_path_drop_masks(
        #             self._path_drop_probabilities[0],
        #             self._path_drop_probabilities[1],
        #             random_values)                                     # 以一种随机性来确定哪一部分留下来
        #
        #
        #         # 选择一种数据输入方式
        #         img_proposal_input = tf.multiply(img_proposal_input,
        #                                          img_mask)               # "rpn_path_drop/Mul:0", shape=(1, 240, 795, 1)
        #
        #         bev_proposal_input = tf.multiply(bev_proposal_input,
        #                                          bev_mask)               # "rpn_path_drop/Mul_1:0", shape=(1, 350, 400, 1)
        #
        #         # print(img_proposal_input)
        #         # print(bev_proposal_input)
        #
        #         # 一种选取处理融合数据的方案
        #         self.img_path_drop_mask = img_mask
        #         self.bev_path_drop_mask = bev_mask
        #
        #
        #         # Overwrite the division factor
        #         fusion_mean_div_factor = img_mask + bev_mask      # 没有0 0的组合,设置的一种组合方式


        with tf.variable_scope('proposal_roi_pooling'):

            with tf.variable_scope('box_indices'):
                def get_box_indices(boxes):
                    proposals_shape = boxes.get_shape().as_list()
                    if any(dim is None for dim in proposals_shape):
                        proposals_shape = tf.shape(boxes)
                    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                    multiplier = tf.expand_dims(
                        tf.range(start=0, limit=proposals_shape[0]), 1)
                    return tf.reshape(ones_mat * multiplier, [-1])

                bev_boxes_norm_batches = tf.expand_dims(
                    self._bev_anchors_norm_pl, axis=0)               # shape (1, ?, 4)

                # These should be all 0's since there is only 1 image
                tf_box_indices = get_box_indices(bev_boxes_norm_batches)   # 0

            # Do ROI Pooling on BEV
            bev_proposal_rois = tf.image.crop_and_resize(
                bev_proposal_input,                                  # 选取的bev  (1, 350, 400, 1)
                self._bev_anchors_norm_pl,                           # (?, 4)
                tf_box_indices,
                self._proposal_roi_crop_size)                        # shape=(?, 3, 3, 1)
            # # Do ROI Pooling on image
            # img_proposal_rois = tf.image.crop_and_resize(
            #     img_proposal_input,                                  # (1, 240, 795, 1)
            #     self._img_anchors_norm_pl,                           # (?, 4)
            #     tf_box_indices,
            #     self._proposal_roi_crop_size)                        # shape=(?, 3, 3, 1)


        # with tf.variable_scope('proposal_roi_fusion'):
        #     rpn_fusion_out = None
        #     if self._fusion_method == 'mean':
        #         tf_features_sum = tf.add(bev_proposal_rois, img_proposal_rois)
        #         rpn_fusion_out = tf.divide(tf_features_sum,
        #                                    fusion_mean_div_factor)    # shape=(?, 3, 3, 1)
        #
        #     elif self._fusion_method == 'concat':
        #         rpn_fusion_out = tf.concat(
        #             [bev_proposal_rois, img_proposal_rois], axis=3)
        #     else:
        #         raise ValueError('Invalid fusion method', self._fusion_method)



        # TODO: move this section into an separate AnchorPredictor class
        with tf.variable_scope('anchor_predictor', 'ap', [bev_proposal_rois]):
            tensor_in = bev_proposal_rois                # shape=(?, 3, 3, 1)

            # Parse rpn layers config
            layers_config = self._config.layers_config.rpn_config
            l2_weight_decay = layers_config.l2_weight_decay          # 0.0005

            if l2_weight_decay > 0:
                weights_regularizer = slim.l2_regularizer(l2_weight_decay)
            else:
                weights_regularizer = None

            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=weights_regularizer):
                # Use conv2d instead of fully_connected layers.
                cls_fc6 = slim.conv2d(tensor_in,
                                      layers_config.cls_fc6,
                                      self._proposal_roi_crop_size,
                                      padding='VALID',
                                      scope='cls_fc6')        # (?, 1, 1, 256)

                cls_fc6_drop = slim.dropout(cls_fc6,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='cls_fc6_drop')     # (?, 1, 1, 256)     # 0.5

                cls_fc7 = slim.conv2d(cls_fc6_drop,
                                      layers_config.cls_fc7,
                                      [1, 1],
                                      scope='cls_fc7')        # 256  # (?, 1, 1, 256)

                cls_fc7_drop = slim.dropout(cls_fc7,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='cls_fc7_drop')        # (?, 1, 1, 256)

                cls_fc8 = slim.conv2d(cls_fc7_drop,
                                      2,
                                      [1, 1],
                                      activation_fn=None,
                                      scope='cls_fc8')                   # (?, 1, 1, 2)

                objectness = tf.squeeze(
                    cls_fc8, [1, 2],
                    name='cls_fc8/squeezed')  # shape=(?, 2)  在特定维度上去掉维数为一的维度
                # print(objectness)

                # Use conv2d instead of fully_connected layers.
                reg_fc6 = slim.conv2d(tensor_in,
                                      layers_config.reg_fc6,
                                      self._proposal_roi_crop_size,
                                      padding='VALID',
                                      scope='reg_fc6')              # (?, 1, 1, 256)

                reg_fc6_drop = slim.dropout(reg_fc6,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='reg_fc6_drop')   # (?, 1, 1, 256)

                reg_fc7 = slim.conv2d(reg_fc6_drop,
                                      layers_config.reg_fc7,
                                      [1, 1],
                                      scope='reg_fc7')              # (?, 1, 1, 256)

                reg_fc7_drop = slim.dropout(reg_fc7,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='reg_fc7_drop')    # (?, 1, 1, 256)

                reg_fc8 = slim.conv2d(reg_fc7_drop,
                                      6,
                                      [1, 1],
                                      activation_fn=None,
                                      scope='reg_fc8')               # (?, 1, 1, 6)

                offsets = tf.squeeze(
                    reg_fc8, [1, 2],
                    name='reg_fc8/squeezed')    # shape=(?, 6)
                # print(offsets)

        # Histogram summaries
        with tf.variable_scope('histograms_feature_extractor'):
            with tf.variable_scope('bev_vgg'):
                for end_point in self.bev_end_points:
                    tf.summary.histogram(
                        end_point, self.bev_end_points[end_point])

            # with tf.variable_scope('img_vgg'):
            #     for end_point in self.img_end_points:
            #         tf.summary.histogram(
            #             end_point, self.img_end_points[end_point])

        with tf.variable_scope('histograms_rpn'):
            with tf.variable_scope('anchor_predictor'):
                fc_layers = [cls_fc6, cls_fc7, cls_fc8, objectness,
                             reg_fc6, reg_fc7, reg_fc8, offsets]
                for fc_layer in fc_layers:
                    # fix the name to avoid tf warnings
                    tf.summary.histogram(fc_layer.name.replace(':', '_'),
                                         fc_layer)

        # Return the proposals
        with tf.variable_scope('proposals'):
            anchors = self.placeholders[self.PL_ANCHORS]    # (?, 6)

            # Decode anchor regression offsets
            with tf.variable_scope('decoding'):
                regressed_anchors = anchor_encoder.offset_to_anchor(
                        anchors, offsets)                   # (?, 6)

            with tf.variable_scope('bev_projection'):
                _, bev_proposal_boxes_norm = anchor_projector.project_to_bev(
                    regressed_anchors, self._bev_extents)            # (?, 4)

            with tf.variable_scope('softmax'):
                objectness_softmax = tf.nn.softmax(objectness)       # (?, 2)

            with tf.variable_scope('nms'):
                objectness_scores = objectness_softmax[:, 1]

                # Do NMS on regressed anchors
                top_indices = tf.image.non_max_suppression(
                    bev_proposal_boxes_norm, objectness_scores,
                    max_output_size=self._nms_size,                 # 1024
                    iou_threshold=self._nms_iou_thresh)             # (?,)

                top_anchors = tf.gather(regressed_anchors, top_indices)   # (?, 6)   提取前n个较高的top_indices 选取的下标
                top_objectness_softmax = tf.gather(objectness_scores,
                                                   top_indices)           # (?,)     选取的objectness
                # top_offsets = tf.gather(offsets, top_indices)
                # top_objectness = tf.gather(objectness, top_indices)

        # Get mini batch
        all_ious_gt = self.placeholders[self.PL_ANCHOR_IOUS]            # (?,)
        all_offsets_gt = self.placeholders[self.PL_ANCHOR_OFFSETS]      # (?, 6)
        all_classes_gt = self.placeholders[self.PL_ANCHOR_CLASSES]      # (?,)

        with tf.variable_scope('mini_batch'):
            mini_batch_utils = self.dataset.kitti_utils.mini_batch_utils
            mini_batch_mask, _ = \
                mini_batch_utils.sample_rpn_mini_batch(all_ious_gt)    # (?,)

        # ROI summary images
        rpn_mini_batch_size = \
            self.dataset.kitti_utils.mini_batch_utils.rpn_mini_batch_size    # 512
        with tf.variable_scope('bev_rpn_rois'):
            mb_bev_anchors_norm = tf.boolean_mask(self._bev_anchors_norm_pl,
                                                  mini_batch_mask)            # (?, 4)
            mb_bev_box_indices = tf.zeros_like(
                tf.boolean_mask(all_classes_gt, mini_batch_mask),
                dtype=tf.int32)                                               # (?,)

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            bev_input_rois = tf.image.crop_and_resize(
                self._bev_preprocessed,
                mb_bev_anchors_norm,
                mb_bev_box_indices,
                (32, 32))                                                   # (?, 32, 32, 6)

            bev_input_roi_summary_images = tf.split(
                bev_input_rois, self._bev_depth, axis=3)                    # (?, 32, 32, 1)
            tf.summary.image('bev_rpn_rois',
                             bev_input_roi_summary_images[-1],
                             max_outputs=rpn_mini_batch_size)

        # with tf.variable_scope('img_rpn_rois'):
        #     # ROIs on image input
        #     mb_img_anchors_norm = tf.boolean_mask(self._img_anchors_norm_pl,
        #                                           mini_batch_mask)             # (?, 4)
        #     mb_img_box_indices = tf.zeros_like(
        #         tf.boolean_mask(all_classes_gt, mini_batch_mask),
        #         dtype=tf.int32)                                                # (?,)
        #
        #     # Do test ROI pooling on mini batch
        #     img_input_rois = tf.image.crop_and_resize(
        #         self._img_preprocessed,
        #         mb_img_anchors_norm,
        #         mb_img_box_indices,
        #         (32, 32))                                                    # (?, 32, 32, 3)
        #
        #     tf.summary.image('img_rpn_rois',
        #                      img_input_rois,
        #                      max_outputs=rpn_mini_batch_size)

        # Ground Truth Tensors
        with tf.variable_scope('one_hot_classes'):

            # Anchor classification ground truth
            # Object / Not Object
            min_pos_iou = \
                self.dataset.kitti_utils.mini_batch_utils.rpn_pos_iou_range[0]    # 0.5

            objectness_classes_gt = tf.cast(
                tf.greater_equal(all_ious_gt, min_pos_iou),
                dtype=tf.int32)                                                   # (?,)
            objectness_gt = tf.one_hot(
                objectness_classes_gt, depth=2,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=self._config.label_smoothing_epsilon)                   # (?, 2)

        # Mask predictions for mini batch
        with tf.variable_scope('prediction_mini_batch'):
            objectness_masked = tf.boolean_mask(objectness, mini_batch_mask)      # (?, 2)
            offsets_masked = tf.boolean_mask(offsets, mini_batch_mask)            # (?, 6)

        with tf.variable_scope('ground_truth_mini_batch'):
            objectness_gt_masked = tf.boolean_mask(
                objectness_gt, mini_batch_mask)                                   # (?, 2)
            offsets_gt_masked = tf.boolean_mask(all_offsets_gt,
                                                mini_batch_mask)                  # (?, 6)

        # Specify the tensors to evaluate
        predictions = dict()

        # Temporary predictions for debugging
        # predictions['anchor_ious'] = anchor_ious
        # predictions['anchor_offsets'] = all_offsets_gt

        if self._train_val_test in ['train', 'val']:
            # All anchors
            predictions[self.PRED_ANCHORS] = anchors                         # (?, 6)

            # Mini-batch masks
            predictions[self.PRED_MB_MASK] = mini_batch_mask                 # (?,)
            # Mini-batch predictions
            predictions[self.PRED_MB_OBJECTNESS] = objectness_masked         # (?, 2)
            predictions[self.PRED_MB_OFFSETS] = offsets_masked               # (?, 6)

            # Mini batch ground truth
            predictions[self.PRED_MB_OFFSETS_GT] = offsets_gt_masked         # (?, 6)
            predictions[self.PRED_MB_OBJECTNESS_GT] = objectness_gt_masked   # (?, 2)

            # Proposals after nms
            predictions[self.PRED_TOP_INDICES] = top_indices                 # (?,)
            predictions[self.PRED_TOP_ANCHORS] = top_anchors                 # (?, 6)
            predictions[
                self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax   # (?,)

        else:
            # self._train_val_test == 'test'
            predictions[self.PRED_TOP_ANCHORS] = top_anchors                 # (?, 6)
            predictions[
                self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax   # (?,)

        return predictions

    def create_feed_dict(self, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """

        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None:
                raise ValueError('sample_index should be None. Do not load '
                                 'particular samples during train or val')

            # During training/validation, we need a valid sample
            # with anchor info for loss calculation
            sample = None
            anchors_info = []

            valid_sample = False
            while not valid_sample:
                if self._train_val_test == "train":
                    # Get the a random sample from the remaining epoch
                    samples = self.dataset.next_batch(batch_size=1)

                else:  # self._train_val_test == "val"
                    # Load samples in order for validation
                    samples = self.dataset.next_batch(batch_size=1,
                                                      shuffle=False)

                # Only handle one sample at a time for now
                sample = samples[0]
                anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

                # When training, if the mini batch is empty, go to the next
                # sample. Otherwise carry on with found the valid sample.
                # For validation, even if 'anchors_info' is empty, keep the
                # sample (this will help penalize false positives.)
                # We will substitue the necessary info with zeros later on.
                # Note: Training/validating all samples can be switched off.
                train_cond = (self._train_val_test == "train" and       # false
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and          # false
                             self._eval_all_samples)
                if anchors_info or train_cond or eval_cond:
                    valid_sample = True
        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = self.dataset.load_samples([sample_index])
            else:
                samples = self.dataset.next_batch(batch_size=1, shuffle=False)

            # Only handle one sample at a time for now
            sample = samples[0]
            anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

        sample_name = sample.get(constants.KEY_SAMPLE_NAME)
        sample_augs = sample.get(constants.KEY_SAMPLE_AUGS)

        # Get ground truth data
        label_anchors = sample.get(constants.KEY_LABEL_ANCHORS)   # (3,6)
        label_classes = sample.get(constants.KEY_LABEL_CLASSES)   # (3)
        # We only need orientation from box_3d
        label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)  # <class 'tuple'>: (3, 7)

        # Network input data
        image_input = sample.get(constants.KEY_IMAGE_INPUT)  # <class 'tuple'>: (375, 1242, 3)
        bev_input = sample.get(constants.KEY_BEV_INPUT)      # <class 'tuple'>: (700, 800, 6)

        # Image shape (h, w)
        image_shape = [image_input.shape[0], image_input.shape[1]]

        ground_plane = sample.get(constants.KEY_GROUND_PLANE)         # <class 'tuple'>: (4,)
        stereo_calib_p2 = sample.get(constants.KEY_STEREO_CALIB_P2)   # <class 'tuple'>: (3, 4)

        # Fill the placeholders for anchor information
        self._fill_anchor_pl_inputs(anchors_info=anchors_info,
                                    ground_plane=ground_plane,
                                    image_shape=image_shape,
                                    stereo_calib_p2=stereo_calib_p2,
                                    sample_name=sample_name,
                                    sample_augs=sample_augs)

        # this is a list to match the explicit shape for the placeholder
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]

        # Fill in the rest
        self._placeholder_inputs[self.PL_BEV_INPUT] = bev_input
        # self._placeholder_inputs[self.PL_IMG_INPUT] = image_input

        self._placeholder_inputs[self.PL_LABEL_ANCHORS] = label_anchors     # <class 'tuple'>: (3, 6)
        self._placeholder_inputs[self.PL_LABEL_BOXES_3D] = label_boxes_3d   # <class 'tuple'>: (3, 7)
        self._placeholder_inputs[self.PL_LABEL_CLASSES] = label_classes     # <class 'tuple'>: (3,)

        # Sample Info
        # img_idx is a list to match the placeholder shape
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]
        self._placeholder_inputs[self.PL_CALIB_P2] = stereo_calib_p2
        self._placeholder_inputs[self.PL_GROUND_PLANE] = ground_plane

        # Temporary sample info for debugging
        self.sample_info.clear()
        self.sample_info['sample_name'] = sample_name
        self.sample_info['rpn_mini_batch'] = anchors_info

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]     # 感觉这个有问题 为什么不是feed_dict[key]　这样值感觉不对劲

        return feed_dict

    def _fill_anchor_pl_inputs(self,
                               anchors_info,
                               ground_plane,
                               image_shape,
                               stereo_calib_p2,
                               sample_name,
                               sample_augs):
        """
        Fills anchor placeholder inputs with corresponding data

        Args:
            anchors_info: anchor info from mini_batch_utils
            ground_plane: ground plane coefficients
            image_shape: image shape (h, w), used for projecting anchors
            sample_name: name of the sample, e.g. "000001"
            sample_augs: list of sample augmentations
        """

        # Lists for merging anchors info
        all_anchor_boxes_3d = []
        anchors_ious = []
        anchor_offsets = []
        anchor_classes = []

        # Create anchors for each class
        if len(self.dataset.classes) > 1:
            for class_idx in range(len(self.dataset.classes)):
                # Generate anchors for all classes
                grid_anchor_boxes_3d = self._anchor_generator.generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=self._cluster_sizes[class_idx],
                    anchor_stride=self._anchor_strides[class_idx],
                    ground_plane=ground_plane)
                all_anchor_boxes_3d.append(grid_anchor_boxes_3d)
            all_anchor_boxes_3d = np.concatenate(all_anchor_boxes_3d)
        else:
            # Don't loop for a single class
            class_idx = 0
            grid_anchor_boxes_3d = self._anchor_generator.generate(
                area_3d=self._area_extents,
                anchor_3d_sizes=self._cluster_sizes[class_idx],
                anchor_stride=self._anchor_strides[class_idx],
                ground_plane=ground_plane)
            all_anchor_boxes_3d = grid_anchor_boxes_3d

        # Filter empty anchors
        # Skip if anchors_info is []
        sample_has_labels = True
        if self._train_val_test in ['train', 'val']:
            # Read in anchor info during training / validation
            if anchors_info:
                anchor_indices, anchors_ious, anchor_offsets, \
                    anchor_classes = anchors_info

                anchor_boxes_3d_to_use = all_anchor_boxes_3d[anchor_indices]
            else:
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if train_cond or eval_cond:
                    sample_has_labels = False
        else:
            sample_has_labels = False

        if not sample_has_labels:
            # During testing, or validation with no anchor info, manually
            # filter empty anchors
            # TODO: share voxel_grid_2d with BEV generation if possible
            voxel_grid_2d = \
                self.dataset.kitti_utils.create_sliced_voxel_grid_2d(
                    sample_name, self.dataset.bev_source,
                    image_shape=image_shape)

            # Convert to anchors and filter
            anchors_to_use = box_3d_encoder.box_3d_to_anchor(
                all_anchor_boxes_3d)
            empty_filter = anchor_filter.get_empty_anchor_filter_2d(
                anchors_to_use, voxel_grid_2d, density_threshold=1)

            anchor_boxes_3d_to_use = all_anchor_boxes_3d[empty_filter]

        # Convert lists to ndarrays
        anchor_boxes_3d_to_use = np.asarray(anchor_boxes_3d_to_use)
        anchors_ious = np.asarray(anchors_ious)
        anchor_offsets = np.asarray(anchor_offsets)
        anchor_classes = np.asarray(anchor_classes)

        # Flip anchors and centroid x offsets for augmented samples
        if kitti_aug.AUG_FLIPPING in sample_augs:
            anchor_boxes_3d_to_use = kitti_aug.flip_boxes_3d(
                anchor_boxes_3d_to_use, flip_ry=False)
            if anchors_info:
                anchor_offsets[:, 0] = -anchor_offsets[:, 0]

        # Convert to anchors
        anchors_to_use = box_3d_encoder.box_3d_to_anchor(
            anchor_boxes_3d_to_use)
        num_anchors = len(anchors_to_use)

        # Project anchors into bev
        bev_anchors, bev_anchors_norm = anchor_projector.project_to_bev(
            anchors_to_use, self._bev_extents)

        # Project box_3d anchors into image space
        img_anchors, img_anchors_norm = \
            anchor_projector.project_to_image_space(
                anchors_to_use, stereo_calib_p2, image_shape)

        # Reorder into [y1, x1, y2, x2] for tf.crop_and_resize op
        self._bev_anchors_norm = bev_anchors_norm[:, [1, 0, 3, 2]]
        self._img_anchors_norm = img_anchors_norm[:, [1, 0, 3, 2]]

        # Fill in placeholder inputs
        self._placeholder_inputs[self.PL_ANCHORS] = anchors_to_use

        # If we are in train/validation mode, and the anchor infos
        # are not empty, store them. Checking for just anchors_ious
        # to be non-empty should be enough.
        if self._train_val_test in ['train', 'val'] and \
                len(anchors_ious) > 0:
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = anchors_ious
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = anchor_offsets
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = anchor_classes

        # During test, or val when there is no anchor info
        elif self._train_val_test in ['test'] or \
                len(anchors_ious) == 0:
            # During testing, or validation with no gt, fill these in with 0s
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = \
                np.zeros(num_anchors)
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = \
                np.zeros([num_anchors, 6])
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = \
                np.zeros(num_anchors)
        else:
            raise ValueError('Got run mode {}, and non-empty anchor info'.
                             format(self._train_val_test))

        self._placeholder_inputs[self.PL_BEV_ANCHORS] = bev_anchors
        self._placeholder_inputs[self.PL_BEV_ANCHORS_NORM] = \
            self._bev_anchors_norm
        self._placeholder_inputs[self.PL_IMG_ANCHORS] = img_anchors
        self._placeholder_inputs[self.PL_IMG_ANCHORS_NORM] = \
            self._img_anchors_norm

    def loss(self, prediction_dict):

        # these should include mini-batch values only
        objectness_gt = prediction_dict[self.PRED_MB_OBJECTNESS_GT]
        offsets_gt = prediction_dict[self.PRED_MB_OFFSETS_GT]

        # Predictions
        with tf.variable_scope('rpn_prediction_mini_batch'):
            objectness = prediction_dict[self.PRED_MB_OBJECTNESS]
            offsets = prediction_dict[self.PRED_MB_OFFSETS]

        with tf.variable_scope('rpn_losses'):
            with tf.variable_scope('objectness'):
                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                objectness_loss = cls_loss(objectness,
                                           objectness_gt,
                                           weight=cls_loss_weight)

                with tf.variable_scope('obj_norm'):
                    # normalize by the number of anchor mini-batches
                    objectness_loss = objectness_loss / tf.cast(
                        tf.shape(objectness_gt)[0], dtype=tf.float32)
                    tf.summary.scalar('objectness', objectness_loss)

            with tf.variable_scope('regression'):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                anchorwise_localization_loss = reg_loss(offsets,
                                                        offsets_gt,
                                                        weight=reg_loss_weight)
                masked_localization_loss = \
                    anchorwise_localization_loss * objectness_gt[:, 1]
                localization_loss = tf.reduce_sum(masked_localization_loss)

                with tf.variable_scope('reg_norm'):
                    # normalize by the number of positive objects
                    num_positives = tf.reduce_sum(objectness_gt[:, 1])
                    # Assert the condition `num_positives > 0`
                    with tf.control_dependencies(
                            [tf.assert_positive(num_positives)]):
                        localization_loss = localization_loss / num_positives
                        tf.summary.scalar('regression', localization_loss)

            with tf.variable_scope('total_loss'):
                total_loss = objectness_loss + localization_loss

        loss_dict = {
            self.LOSS_RPN_OBJECTNESS: objectness_loss,
            self.LOSS_RPN_REGRESSION: localization_loss,
        }

        return loss_dict, total_loss

    def create_path_drop_masks(self,
                               p_img,
                               p_bev,
                               random_values):
        """Determines global path drop decision based on given probabilities.

        Args:
            p_img: A tensor of float32, probability of keeping image branch
            p_bev: A tensor of float32, probability of keeping bev branch
            random_values: A tensor of float32 of shape [3], the results
                of coin flips, values should range from 0.0 - 1.0.

        Returns:
            final_img_mask: A constant tensor mask containing either one or zero  0 or 1
                depending on the final coin flip probability.
            final_bev_mask: A constant tensor mask containing either one or zero  0 or 1
                depending on the final coin flip probability.
        """

        def keep_branch(): return tf.constant(1.0)

        def kill_branch(): return tf.constant(0.0)

        # The logic works as follows:
        # We have flipped 3 coins, first determines the chance of keeping
        # the image branch, second determines keeping bev branch, the third
        # makes the final decision in the case where both branches were killed
        # off, otherwise the initial img and bev chances are kept.

        img_chances = tf.case([(tf.less(random_values[0], p_img),
                                keep_branch)], default=kill_branch)   # < p_img时,返回1

        bev_chances = tf.case([(tf.less(random_values[1], p_bev),
                                keep_branch)], default=kill_branch)   # < p_bev时,返回1

        # Decision to determine whether both branches were killed off
        third_flip = tf.logical_or(tf.cast(img_chances, dtype=tf.bool),
                                   tf.cast(bev_chances, dtype=tf.bool))   # 两个都是假时返回0
        third_flip = tf.cast(third_flip, dtype=tf.float32)

        # Make a second choice, for the third case
        # Here we use a 50/50 chance to keep either image or bev
        # If its greater than 0.5, keep the image
        img_second_flip = tf.case([(tf.greater(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)
        # If its less than or equal to 0.5, keep bev
        bev_second_flip = tf.case([(tf.less_equal(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)

        # Use lambda since this returns another condition and it needs to
        # be callable
        final_img_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: img_chances)],
                                 default=lambda: img_second_flip)

        final_bev_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: bev_chances)],
                                 default=lambda: bev_second_flip)

        return final_img_mask, final_bev_mask



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    # Defaults
    default_pipeline_config_path = avod.root_dir() + \
                                   '/configs/avod_cars_example.config'
    default_data_split = 'train'
    default_device = '0'

    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        default=default_pipeline_config_path,
                        help='Path to the pipeline config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=default_data_split,
                        help='Data split for training')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default=default_device,
                        help='CUDA device id')

    args = parser.parse_args()

    # Parse pipeline config
    model_config, train_config, _, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.pipeline_config_path, is_training=True)

    # print(model_config)
    # print(train_config)
    # print(dataset_config)

    # Overwrite data split
    dataset_config.data_split = args.data_split

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    train_val_test = 'train'
    model_name = model_config.model_name

    with tf.Graph().as_default():
        if model_name == 'avod_model':

            model = RpnModel(model_config,
                             train_val_test=train_val_test,
                             dataset=dataset)
            trainer.train(model, train_config)
            # predictions = model.build()

            # with tf.Session() as sess:
            #     print(sess.run(predictions))







