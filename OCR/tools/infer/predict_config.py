import os

use_gpu = True
det_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              './model/det_infer.pth')
rec_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              './model/rec_infer.pth')
det_limit_side_len = 960
det_limit_type = 'max'
det_db_thresh = 0.3
det_db_box_thresh = 0.5
det_db_unclip_ratio = 1.6
max_batch_size = 10
rec_algorithm = 'CRNN'
rec_image_inverse = True
rec_char_type = 'ch'
rec_image_shape = '3, 32, 320'
rec_batch_num = 6
max_text_length = 25
use_space_char = True
drop_score = 0.5
limited_max_width = 1280
limited_min_width = 16
rec_char_dict_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  'pytorchocr/utils/ocr_keys')
image_dir = ''