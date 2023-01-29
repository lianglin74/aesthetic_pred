import os

from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #

_C.MODEL = CN()
#arch
_C.MODEL.ARCH = 'resnet18'
#torch_arch, resnet_two_fcs, msft_caffe, attention, classifier_attention, reset_two_models, resnet10, MobileNetV3, resnet_spacial_attention, resnet_multiheads, efficientnet-b0, cls_resnet, cls_vit, cls_vit_multiheads
_C.MODEL.MODEL_TYPE = 'torch_arch'
# pred activation
_C.MODEL.PRED_ACTIVATION = 'none'
#pretrained
_C.MODEL.USE_PRETRAINED_TORCH = False
_C.MODEL.PRETRAINED_CLASS_NUM = -1
#modelfile
_C.MODEL.WEIGHT = ''
#modelfile_use_all
_C.MODEL.WEIGHT_USE_ALL = False
#msft_caffe_folder
_C.MODEL.MSFT_CAFFE_FOLDER=''
# msft cafe model pooling method: roi_max, roi_avg, global_ave, roi_end, none
_C.MODEL.MSFT_CAFFE_POOLING = 'none'
#cls_resnet config file
_C.MODEL.CLS_RESNET_CONFIG=''
#clip pretrained model file
_C.MODEL.CLIP_VIT_MODEL=''
_C.MODEL.CLIP_PROJ_NONE=False
#vit pretrained model type
_C.MODEL.TIMM_MODEL_TYPE=''
#vit pretrained model file
_C.MODEL.TIMM_MODEL_FILE=''
_C.MODEL.CLIP_NORM_IMAGE_FEATURE = True
_C.MODEL.QUERY2LABEL_CONFIG=''

#efficientnet batch_norm_momentum
_C.MODEL.EFFICIENTNET_BN_MOMENTUM=0.99
_C.MODEL.EFFICIENTNET_BN_EPS=1e-3
_C.MODEL.EFFICIENTNET_OUT_CHANNELS=1280
#bn_init_zero_r
_C.MODEL.BN_INIT_ZERO_R = False
#train_all, fixfeature, fixpartialfeature
_C.MODEL.FREEZE_WEIGHTS_TYPE = 'train_all'
# this parameter provides a flexible way to select which
# parameters should be frozen by specifying a regular expression, so that
# any weight that matches it will be frozen
# examples:
#   ".*backbone.*" --> to freeze backbone
#   ".*(backbone|rpn).*" --> to freeze backbone and rpn
_C.MODEL.FREEZE_WEIGHTS_REGEXP = ""  # if empty string this parameter is ignored
# the classifier initial weight file
_C.MODEL.CLASSIFIER_INIT_WEIGHT = ''
# model classifier init type
_C.MODEL.CLASSIFIER_INIT_TYPE = 'bias_zero'
# embed dimension
_C.MODEL.EMBED_DIM = -1
_C.MODEL.DROP_PATH = 0.1
#DistributedDataParallel
_C.MODEL.FIND_UNUSED_PARAMETERS = True
_C.MODEL.CHANGE_HEAD_OUTPUT_DIM = False

_C.MODEL.VERBOSE = False

#BEIT
_C.BEIT = CN()
_C.BEIT.LAYER_DECAY = 0.85
_C.BEIT.DROP_PATH = 0.1
_C.BEIT.CLIP_GRAD = -1.0
_C.BEIT.SAVE_CKPT_FREQ = 1
_C.BEIT.UPDATE_FREQ = 1
_C.BEIT.USE_TAGGER_RECIPE = False
_C.BEIT.NO_AMP = False
_C.BEIT.INIT_SCALE = 0.001

#VIT
_C.VIT = CN()
_C.VIT.DROP_PATH = 0.1

#cls_vit_with_labels
_C.VIT_WITH_LABELS = CN()
_C.VIT_WITH_LABELS.MAX_LABELS_LENGTH = 210
_C.VIT_WITH_LABELS.MASK_ALL_LABELS_PROB = 0.5
_C.VIT_WITH_LABELS.MASKED_PROB_MIN = 0.15
_C.VIT_WITH_LABELS.MASKED_PROB_MAX = 0.5
_C.VIT_WITH_LABELS.MAX_PRED_ITERATIONS = 3
_C.VIT_WITH_LABELS.MAX_PRED_LABELS_LENGTH = 50
_C.VIT_WITH_LABELS.SCORE_THRESHOLD = 0.0
_C.VIT_WITH_LABELS.CHECK_MATRIX = False
_C.VIT_WITH_LABELS.NONLINEAR_HEAD = False
_C.VIT_WITH_LABELS.NORMALIZE_MASKED_TOKEN_LOSS = False
_C.VIT_WITH_LABELS.HIDDEN_ACT = 'gelu'

#cls_vit_multiscale_tokens
_C.VIT_MULTISCALE_TOKENS = CN()
_C.VIT_MULTISCALE_TOKENS.MULTI_SCALES_COUNT = 2
_C.VIT_MULTISCALE_TOKENS.ATTENTION_WINDOW_SIZE = [-1,-1]
_C.VIT_MULTISCALE_TOKENS.CROSS_ATTENTION_FINE_TO_COARSE_WINDOW_SIZE = [-1]
_C.VIT_MULTISCALE_TOKENS.CROSS_ATTENTION_COARSE_TO_FINE_WINDOW_SIZE = [-1]
_C.VIT_MULTISCALE_TOKENS.USE_ATTENTION_MASK = False
_C.VIT_MULTISCALE_TOKENS.CHECK_MATRIX = False
_C.VIT_MULTISCALE_TOKENS.IMAGE_INTERPOLATION = 'bilinear'

#query2label
_C.QUERY2LABEL = CN()
_C.QUERY2LABEL.gamma_pos = 0.0
_C.QUERY2LABEL.gamma_neg = 2.0
_C.QUERY2LABEL.cutout = False
_C.QUERY2LABEL.n_holes = 1
_C.QUERY2LABEL.cut_fact = 0.5
_C.QUERY2LABEL.hidden_dim = 2048
_C.QUERY2LABEL.dim_feedforward = 8192
_C.QUERY2LABEL.keep_input_proj = False
_C.QUERY2LABEL.enc_layers = 1
_C.QUERY2LABEL.dec_layers = 2
_C.QUERY2LABEL.nheads = 4
_C.QUERY2LABEL.early_stop = False
_C.QUERY2LABEL.amp = False
_C.QUERY2LABEL.dtgfl = False
_C.QUERY2LABEL.backbone_path = 'none'
#['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
_C.QUERY2LABEL.clip_model = 'RN101'
_C.QUERY2LABEL.clip_model_merge_class_token = 'drop'
_C.QUERY2LABEL.clip_model_text_query_type = 'word'
_C.QUERY2LABEL.clip_model_image_embedding_type = 'feature_map_no_proj' #Or 'feature_map_no_proj_normalize', 'feature_map_proj_normalize'
_C.QUERY2LABEL.clip_model_text_embedding_type = 'no_normalize' #or 'normalize', 'normalize_then_logscale'
_C.QUERY2LABEL.clip_model_clip_logit_scale = False
_C.QUERY2LABEL.mlp_ratio = 4.0
_C.QUERY2LABEL.add_pos_embedding = True

#MMGPT
_C.GIT = CN()
_C.GIT.IMAGE_ENCODER_CONFIG = 'none'
_C.GIT.PRETRAINED_WEIGHTS = 'none'
_C.GIT.MAX_TOKEN_LENGTH = 40
_C.GIT.SHUFFLE_TAG = False

# ---------------------------------------------------------------------------- #
# DISTILLATION
# ---------------------------------------------------------------------------- #
_C.DISTILLATION = CN()
_C.DISTILLATION.USE = False
_C.DISTILLATION.TEACHER_LAMBDA = 0.5
_C.DISTILLATION.TEACHER_LAMBDA_MULTIHEADS = []
# 'one_side' means only weighting the tearcher loss term, 'two_side' means: (1-a)*Ls + a*Lt, Ls is the stude
_C.DISTILLATION.TEACHER_LAMBDA_TYPE = 'one_side'
_C.DISTILLATION.T = 5.0
_C.DISTILLATION.MASK_HARD_NEG = False

# for generating hard label from teacher
_C.DISTILLATION.TEACHER_HARD_LABEL_THRESHOLD = 0.0
_C.DISTILLATION.TEACHER_HARD_LABEL_LOSS_WEIGHT = 0.5
_C.DISTILLATION.TEACHER_HARD_LABEL_WITH_UNKNOWN = False

_C.DISTILLATION.MODEL = CN()
#arch
_C.DISTILLATION.MODEL.ARCH = 'resnet18'
#torch_arch, resnet_two_fcs, msft_caffe, attention, classifier_attention, reset_two_models, resnet10, MobileNetV3, resnet_spacial_attention, resnet_multiheads, efficientnet-b0, cls_resnet
_C.DISTILLATION.MODEL.MODEL_TYPE = 'torch_arch'
#pretrained
_C.DISTILLATION.MODEL.USE_PRETRAINED_TORCH = False
_C.DISTILLATION.MODEL.PRETRAINED_CLASS_NUM = -1
#modelfile
_C.DISTILLATION.MODEL.WEIGHT = ''
#modelfile_use_all
_C.DISTILLATION.MODEL.WEIGHT_USE_ALL = False
#msft_caffe_folder
_C.DISTILLATION.MODEL.MSFT_CAFFE_FOLDER=''
# msft cafe model pooling method: roi_max, roi_avg, global_ave, roi_end, none
_C.DISTILLATION.MODEL.MSFT_CAFFE_POOLING = 'none'
#cls_resnet config file
_C.DISTILLATION.MODEL.CLS_RESNET_CONFIG=''
#timm model type
_C.DISTILLATION.MODEL.TIMM_MODEL_TYPE=''
_C.DISTILLATION.MODEL.TIMM_MODEL_FILE=''
#efficientnet batch_norm_momentum
_C.DISTILLATION.MODEL.EFFICIENTNET_BN_MOMENTUM=0.99
_C.DISTILLATION.MODEL.EFFICIENTNET_BN_EPS=1e-3
#train_all, fixfeature, fixpartialfeature
_C.DISTILLATION.MODEL.FREEZE_WEIGHTS_TYPE = 'train_all'
#distillation loss type
_C.DISTILLATION.LOSS_TYPE = 'SOFTMAX_KL'
#distillation start epoch
_C.DISTILLATION.START_EPOCH = 0

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = []
_C.DATASETS.TEST = []
_C.DATASETS.TEST_LABEL = []
_C.DATASETS.LABELMAP_FILE = 'none'
# The head index that the yaml in _C.DATASETS.TRAIN corresponds to
_C.DATASETS.TRAIN_HEAD_INDICES = []
# Training data classes frequency file
_C.DATASETS.TRAIN_CLASSES_FREQ = 'none'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
#j
_C.DATALOADER.NUM_WORDERS = 4
#-b, batch-size per gpu
_C.DATALOADER.BATCH_SIZE = 64
#train_no_shuffle
_C.DATALOADER.TRAIN_NO_SHUFFLE = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
#crop size
_C.INPUT.CROP_SIZE = 224
#train_image_size
_C.INPUT.TRAIN_IMAGE_SIZE = -1
#bgr_tagging, rgb_tagging, rgb_product, msft_caffe, attention, product_canonical, product_keyframe, product_target_domain, product_imagenet_augmentation, \
# bgr_tagging_cv_bilinear, bgr_tagging_cv_smooth_bilinear, bgr_tagging_cv_RandomResizeCrop_bilinear, bgr_tagging_cv_RandomResizeCrop_smooth_bilinear, bgr_tagging_pil_box \
# msft_caffe_padding, bgr_tagging_randcut, rgb_tagging_randcut
_C.INPUT.DATA_TRANS_TYPE = 'none'
# resize interpolation method
_C.INPUT.RESIZE_INTERPOLATION = 'bilinear'
#color_aug_para
_C.INPUT.COLOR_AUG_PARA = [0.0, 0.0, 0.0, 0.0]
#rotation_aug_para
_C.INPUT.ROTATION_AUG_PARA = 0.0
#affine shear
_C.INPUT.AFFINE_SHEAR_DEGREE = [0.0, 0.0, 0.0, 0.0]
#random crop scale
_C.INPUT.CROP_SCALE = [0.08, 1.0]
#random crop ratio
_C.INPUT.CROP_RATIO = [3. / 4., 4. / 3.]
_C.INPUT.CUTOUT_FACTOR = 2.0
_C.INPUT.CUTOUT_P = 0.0
_C.INPUT.RANDAUG_CONFIG_STR = 'rand-m9-mstd0.5'
_C.INPUT.RANDAUG_TRANS_CONST = 0.45
#gaussian blur
_C.INPUT.GAUSSIAN_BlUR_MAX_RADIUS = 3
_C.INPUT.GAUSSIAN_BlUR_ADD_PROB = 0.0
#Gaussian noise
_C.INPUT.GAUSSIAN_NOISE_STD = 0.0
_C.INPUT.GAUSSIAN_NOISE_ADD_PROB = 0.0
#cutmix
_C.INPUT.CUTMIX_BETA = 1.0
_C.INPUT.CUTMIX_PROB = 0.0
#MSFT transform parameters
_C.INPUT.MSFT_IMAGE_SIZE = 256
_C.INPUT.MSFT_CANVAS_SIZE = 512
_C.INPUT.MSFT_SQUARE_CROP = False
_C.INPUT.CLIP_RANDOM_CENTER_CROP_PADDING = 32

# ---------------------------------------------------------------------------- #
# Criterion
# ---------------------------------------------------------------------------- #
_C.CRITERION = CN()
#SingleLabelSoftmax, SingleLabelSoftmaxBalance, MultiLabelSoftmax, Sigmoid, SigmoidBalance, TreeLoss, \
# SigmoidBalanceGtNeg, MultilabelSoftmaxGtNeg, MultiLabelSoftmaxCos, IBCEWithLogitsNegLoss, IBCEWithLogitsNegLossGtNeg, MultilabelRankLoss, MultilabelRankLossGtNeg
# FocalLoss, SmoothFocalLoss, SmoothFocalLossGtNeg, MultiLabelSoftmaxDistillation, ConcurrentSoftmax, SmoothLabelCrossEntropyLoss
_C.CRITERION.TYPE = 'SingleLabelSoftmax'
# reduction type
_C.CRITERION.REDUCTION = 'mean'
#neg
_C.CRITERION.SIGMOID_CLASSWISE_NEG_SAMPLES_WEIGHT = ''
_C.CRITERION.SIGMOID_CLASSWISE_POS_SAMPLES_WEIGHT = ''
#
_C.CRITERION.CLASS_WEIGHT = ''
#label_smoothing_value
_C.CRITERION.LABEL_SMOOTHING_VALUE = 0.0
#focalloss alpha
_C.CRITERION.FOCALLOSS_ALPHA = 0.25
#focalloss gamma
_C.CRITERION.FOCALLOSS_GAMMA = 2.0
#focalloss sigmoid_pred no gradient
_C.CRITERION.FOCALLOSS_SIGMOID_PRED_NO_GRAD = False
#asymmetrical focalloss
_C.CRITERION.FOCALLOSS_GAMMA_POS = 0.0
_C.CRITERION.FOCALLOSS_GAMMA_NEG = 1.0
_C.CRITERION.FOCALLOSS_NEG_WEIGHT = 1.0
#smoothfocalloss pos probability
_C.CRITERION.SMOOTH_BCE_POS_PROB = 0.8
_C.CRITERION.SMOOTH_BCE_NEG_PROB = 0.2
_C.CRITERION.SMOOTH_FOCALLOSS_CLAMP_MIN = 2.0
_C.CRITERION.SMOOTH_FOCALLOSS_CLAMP_MAX = 2.0
# The epochs to stablize
_C.CRITERION.SMOOTH_BCE_INIT_EPOCHS = 1000
_C.CRITERION.SMOOTH_BCE_POS_MIN_PROB = 0.0
_C.CRITERION.SMOOTH_BCE_NEG_MAX_PROB = 1.0

#sigmoid ignore easy pos and neg samples
_C.CRITERION.SIGMOID_EASY_NEG_SCORE_THR = -2.0
_C.CRITERION.SIGMOID_EASY_POS_SCORE_THR = 2.0
#sigmoid loss normalization type: norm_by_weight, norm_by_weight_sum
_C.CRITERION.SIGMOID_LOSS_NORMALIZATION_TYPE = 'norm_by_weight'

#multiheads: weight of each head's loss
_C.CRITERION.DATASETS_LOSS_WEIGHT = []

#concurrent softmax
_C.CRITERION.CONCURRENT_SOFTMAX_INV_CONCURRENT_MATRIX = 'none'

#long-tail learning via logits adjustment
_C.CRITERION.CLASS_PRIOR_W = -1.0

# for distribution balance loss
_C.CRITERION.CLASS_EXPECT_SAMPLE_FREQ = 'none'

# for ava rating regression
_C.CRITERION.AVA_REGRESS_RATING = False
_C.CRITERION.AVA_FocalEMDLoss_TYPE = 'mean_score'
_C.CRITERION.AVA_FocalEMDLoss_WEIGHT_POW_R = 2.0
_C.CRITERION.AVA_MeanAndEMDLoss_MEAN_WEIGHT = 1.0

# ---------------------------------------------------------------------------- #
# For gt_neg loss term
# ---------------------------------------------------------------------------- #
_C.CRITERION.GT_NEG = CN()
#gt_neg_cost_weight
_C.CRITERION.GT_NEG.LOSS_WEIGHT = 0.0
#gt_neg_class_weights_file
_C.CRITERION.GT_NEG.SIGMOID_CLASSWISE_NEG_SAMPLES_WEIGHT=''
_C.CRITERION.GT_NEG.SIGMOID_CLASSWISE_POS_SAMPLES_WEIGHT='' 
#class_weights
_C.CRITERION.GT_NEG.CLASS_WEIGHT = ''
#focalloss gamma
_C.CRITERION.GT_NEG.FOCALLOSS_GAMMA = 0.0
#sigmoid ignore easy pos and neg samples
_C.CRITERION.GT_NEG.SIGMOID_EASY_NEG_SCORE_THR = -2.0
_C.CRITERION.GT_NEG.SIGMOID_EASY_POS_SCORE_THR = 2.0
#smoothfocalloss pos probability
_C.CRITERION.GT_NEG.SMOOTH_BCE_POS_PROB = 1.0
#focalloss gamma
_C.CRITERION.GT_NEG.SMOOTH_BCE_NEG_PROB = 0.0

# ---------------------------------------------------------------------------- #
# Jianfeng's IBCEWithLogitsNegLoss loss
# ---------------------------------------------------------------------------- #
_C.CRITERION.IBCEWithLogitsNegLoss = CN()
_C.CRITERION.IBCEWithLogitsNegLoss.EASY_NEG_SCORE_THR = 0.1
_C.CRITERION.IBCEWithLogitsNegLoss.EASY_POS_SCORE_THR = 0.9
_C.CRITERION.IBCEWithLogitsNegLoss.NEG_POS_RATIO = 1.0
_C.CRITERION.IBCEWithLogitsNegLoss.IGNORE_HARD_NEG_SCORE_THR = 2.0
_C.CRITERION.IBCEWithLogitsNegLoss.IGNORE_HARD_POS_SCORE_THR = -2.0
_C.CRITERION.IBCEWithLogitsNegLoss.GT_NEG_WEIGHT = 1.0


# ---------------------------------------------------------------------------- #
# MultilabelRankLoss
# ---------------------------------------------------------------------------- #
_C.CRITERION.MULTILABEL_RANK_LOSS = CN()
_C.CRITERION.MULTILABEL_RANK_LOSS.HAS_BACKGROUND_CLASS = False
_C.CRITERION.MULTILABEL_RANK_LOSS.WEIGHT_UNKNONW_CLASS = 0.1
_C.CRITERION.MULTILABEL_RANK_LOSS.GT_NEG_MARGIN = 0.0
_C.CRITERION.MULTILABEL_RANK_LOSS.MARGIN = 0.0

# ---------------------------------------------------------------------------- #
# For tree loss
# ---------------------------------------------------------------------------- #
_C.CRITERION.TREE = CN()
#label_tree
_C.CRITERION.TREE.LABEL_TREE = ''
#tree_loss_type: SingleLabelSoftmax, MultiLabelSoftmax, Sigmoid
_C.CRITERION.TREE.LOSS_TYPE = 'MultiLabelSoftmax'
#tree_group_weight_type: batch_norm, equal
_C.CRITERION.TREE.GROUP_WEIGHT_TYPE = 'batch_norm'

# ---------------------------------------------------------------------------- #
# For fc weight cos loss: make confused classes futher away
# ---------------------------------------------------------------------------- #
_C.CRITERION.COS = CN()
_C.CRITERION.COS.CONFUSED_LABELS_FILE = ''
_C.CRITERION.COS.WEIGHT = 0.0
_C.CRITERION.COS.TOPK = -1

# ---------------------------------------------------------------------------- #
# for knowledge distillation
# ---------------------------------------------------------------------------- #
_C.CRITERION.DISTILLATION = CN()
_C.CRITERION.DISTILLATION.TEACHER_T = 5.0
_C.CRITERION.DISTILLATION.ALPHA = 0.1

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# type
_C.SOLVER.TYPE = 'SGD'
#epochs
_C.SOLVER.EPOCHS = 90
#lr
_C.SOLVER.BASE_LR = 0.1
_C.SOLVER.PARAMETER_GROUPS = []
#gamma_lr0
_C.SOLVER.BASE_LR_GAMMA = 0.
#target lr
_C.SOLVER.TARGET_LR = 0.0
#fc_lr_scale
_C.SOLVER.FC_LR_SCALE = 1.0
#momentum
_C.SOLVER.MOMENTUM = 0.9
#weight-decay
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.FC_WEIGHT_DECAY = 1e-4
#no_weight_decay
_C.SOLVER.BN_BIAS_NO_WEIGHT_DECAY = False
# SGD nesterov
_C.SOLVER.SGD_NESTEROV = False
#resume
_C.SOLVER.RESUME_TRAIN = False
#warmup-lr
_C.SOLVER.WARMUP_LR = 0.0
#warmup-epochs
_C.SOLVER.WARMUP_EPOCHS = 0
#warmup-iterations
_C.SOLVER.WARMUP_ITERATIONS = 0
#Auto Mixed Precision Training
_C.SOLVER.MIXED_PRECISION = False
#larc
_C.SOLVER.LARS = False
#remove from weight decay
_C.SOLVER.WITHOUT_WD_MODULE_TYPE_LIST = ['LayerNorm', 'GroupNorm', 'BatchNorm2d', 'bias',]
_C.SOLVER.WITHOUT_WD_PARAMETER_KEY_WORDS_LIST = []
#lr depth decay
_C.SOLVER.TRAIN_LR_DECAY = 1.0
_C.SOLVER.TRAIN_LR_DECAY_END = 1.0

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.SOLVER.OPTIMIZER = CN()
#lr-policy: STEP, MULTISTEP, EXPONENTIAL, PLATEAU, CONSTANT
_C.SOLVER.OPTIMIZER.POLICY = 'step'
#step-size
_C.SOLVER.OPTIMIZER.STEP_SIZE = 30
#milestones
_C.SOLVER.OPTIMIZER.MILESTONES = []
#gamma
_C.SOLVER.OPTIMIZER.GAMMA = 0.1

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.TEST_ONLY = False
_C.TEST.IMAGE_SIZE = 256
_C.TEST.DATA_DIR = []
_C.TEST.MODEL_NAMES = []
_C.TEST.MODEL_EXTENTIONS = '.pth.tar'
_C.TEST.FORCE_PRED = False
#softmax, sigmoid, tree_softmax_multiply, tree_softmax_direct, tree_sigmoid_multiply, tree_softmax_direct, softmax_rank_loss
_C.TEST.PRED_FUNC = 'softmax'
_C.TEST.PRED_LABELMAP = 'none'
#test_trans_type: bgr_tagging, rgb_tagging, rgb_product, rgb_product_center_crop, msft_caffe, bgr_tagging_affine, bgr_tagging_cv_bilinear, bgr_tagging_cv_smooth_bilinear, bgr_tagging_resize_cv_bilinear, bgr_tagging_resize_cv_smooth_bilinear, bgr_tagging_pil_box \
#msft_caffe_padding
_C.TEST.DATA_TRANS_TYPE = 'none'
_C.TEST.SCORE_THRESHOLD = []
_C.TEST.TOPK = -1
#tag, json, both
_C.TEST.OUTPUT_FORMAT = 'both'
#detdeval, mAP_classes, F1
_C.TEST.EVAL_FUNC = 'deteval'
_C.TEST.EVAL_USE_GT_NEG = False
_C.TEST.LABELMAP = 'none' #gt_labelmap: do not use labelmap. none: use model's labelmap
_C.TEST.HEAD_LABELMAP_LIST = []
_C.TEST.MSFT_CAFFE_POOLING = 'train'
_C.TEST.LABELS_STATS = 'none'
_C.TEST.HEAD_THRESHOLD_LIST = []

#output-dir
_C.DATA_DIR = []
_C.OUTPUT_DIR = ''
#prefix
_C.PREFIX = 'none'
_C.PRINT_FREQ = 100

#deepspeed
_C.ENABLE_DEEP_SPEED = False
_C.DEEP_SPEED_zero_opt_stage = 1
_C.GRADIENT_CLIP = 0.0

#torch fp16
_C.ENABLE_TORCH_FP16 = False

#distribution training
_C.TORCH_INIT_PROCESS_GROUP_TIMEOUT_SECS = 0