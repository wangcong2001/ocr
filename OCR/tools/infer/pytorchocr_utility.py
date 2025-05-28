import os
import argparse

# 获取当前目录
__dir__ = os.path.dirname(os.path.abspath(__file__))
__fdir__ = os.path.abspath(os.path.join(__dir__, '../..'))


# def init_args():
#     def argbool(v):
#         return v.lower() in ("true", "t", "1")
#     print(__fdir__)
#     parser = argparse.ArgumentParser()
#     parser.set_defaults(use_gpu=True, type=argbool)
#     parser.add_argument("--image_dir", type=str)
#     parser.add_argument("--det_model_path", type=str,
#                         default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
#                               './model/det_infer.pth'))
#     parser.add_argument("--rec_model_path", type=str,
#                         default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
#                               './model/rec_infer.pth'))
#     parser.set_defaults(det_limit_side_len=960, type=float)
#     parser.set_defaults(det_limit_type='max', type=str)
#
#     parser.set_defaults(det_db_thresh=0.3, type=float)
#     parser.set_defaults(det_db_box_thresh=0.5, type=float)
#     parser.set_defaults(det_db_unclip_ratio=1.6, type=float)
#     parser.set_defaults(max_batch_size=10, type=int)
#
#     parser.set_defaults(rec_algorithm='CRNN', type=str)
#     parser.set_defaults(rec_image_inverse=True, type=argbool)
#     parser.set_defaults(rec_char_type='ch', type=str)
#     parser.set_defaults(rec_image_shape='3, 32, 320', type=str)
#     parser.set_defaults(rec_batch_num=6, type=int)
#
#     parser.set_defaults(max_text_length=25, type=int)
#     parser.set_defaults(use_space_char=True, type=argbool)
#     parser.set_defaults(drop_score=0.5, type=float)
#     parser.set_defaults(limited_max_width=1280, type=int)
#     parser.set_defaults(limited_min_width=16, type=int)
#     parser.add_argument(
#         "--rec_char_dict_path",
#         type=str,
#         default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
#                              'pytorchocr/utils/ocr_keys'))
#     return parser
#
# def parse_args():
#     parser = init_args()
#     return parser.parse_args()
#
# def get_default_config(args):
#     return vars(args)


# 加载模型数据
def LoadConfig(weights_path, ):
    if not os.path.exists(os.path.abspath(weights_path)):
        raise FileNotFoundError('{} is not found.'.format(weights_path))
    weights_basename = os.path.basename(weights_path)
    weights_name = weights_basename.lower()
    if weights_name == 'det_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Backbone': {'name': 'ResNet_det', 'layers': 18, 'disable_se': True},
                          'Neck': {'name': 'DBFPN', 'out_channels': 256},
                          'Head': {'name': 'DBHead', 'k': 50}}
    elif weights_name == 'rec_infer.pth':
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Backbone': {'name': 'ResNet_rec', 'layers': 34},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 256, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}
    return network_config
