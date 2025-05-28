import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
import cv2
import numpy as np
import time

import torch
from OCR.tools.infer.base import Base
import OCR.tools.infer.pytorchocr_utility as utility
from OCR.pytorchocr.utils.img_util import get_image_file_list, get_gif_file
from OCR.pytorchocr.utils import preprocess, transform
from OCR.pytorchocr.postprocess import build_post_process



class TextDetector(Base):
    def __init__(self, args, **kwargs):
        self.args = args
        # 预处理参数
        pre_process = [
            {
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
            }}, 
            {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }},
            {
            'ToCHWImage': None
            },
            {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }}
            ]
        # 后处理参数
        post_params = {}
        post_params['name'] = 'DBPostProcess'    # 后处理方法的名称
        post_params["thresh"] = args.det_db_thresh # 阈值
        post_params["box_thresh"] = args.det_db_box_thresh  # 文本框阈值
        post_params["max_candidates"] = 1000     # 最大候选框数量
        post_params["unclip_ratio"] = args.det_db_unclip_ratio # 裁剪比例

        # 预处理
        self.preprocess = preprocess(pre_process)
        # 后处理
        self.postprocess = build_post_process(post_params)

        # cuda
        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu
        # 权重路径
        self.weights_path = args.det_model_path

        network_config = utility.LoadConfig(self.weights_path)
        super(TextDetector, self).__init__(network_config, **kwargs)


        # 加载权重
        self.net.load_state_dict(torch.load(self.weights_path))
        print('model is load: {}'.format(self.weights_path))
        # 设置推理模式（评估）
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()


    # 按照顺时针顺序对输入的四个点坐标进行排序
    def order_points_clockwise(self, pts):
        # 四个点按照 x 坐标进行排序
        xSorted = pts[np.argsort(pts[:, 0]), :]
        # 从排序后的 xSorted 中取出左边两个点和右边两个点，得到 leftMost 和 rightMost。
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        # 按照 y 坐标对 leftMost得到 tl, bl, tr, br 四个点。
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        # 左上角和左下角
        (tl, bl) = leftMost
        # 右上角和右下角
        (tr, br) = rightMost
        # 返回排序后的四个点坐标
        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect
    
    # 将检测到的文本区域的坐标点限制在图像的边界内
    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    # 过滤检测到的文本区域
    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        detection_boxes_new = []
        for box in dt_boxes:
            # 排序
            box = self.order_points_clockwise(box)
            # 限制坐标点在图像边界内
            box = self.clip_det_res(box, img_height, img_width)
            # 计算文本框的宽度和高度
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            # 过滤掉宽度或高度小于3像素的文本框
            if rect_width <= 3 or rect_height <= 2:
                continue
            detection_boxes_new.append(box)
        # 返回过滤后的文本框
        detection_boxes = np.array(detection_boxes_new)
        return detection_boxes

    def __call__(self, img):
        # 保存原始图像
        origin_image = img.copy()
        # 预处理
        data = {'image': img}
        # transform 执行预处理
        data = transform(data, self.preprocess)
        # 获取图像和图像的形状
        img, shape_list = data
        # 如果图像为空，返回空
        if img is None:
            return None, 0
        # 将图像扩展一个维度
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        # 保存图像
        img = img.copy()
        starttime = time.time()
        # 计算
        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu:
                inp = inp.cuda()
            outputs = self.net(inp)
        # 获取预测结果
        preds = {}
        preds['maps'] = outputs['maps'].cpu().numpy()

        # 后处理
        post_result = self.postprocess(preds, shape_list)
        # 获取文本框
        detection_boxes = post_result[0]['points']
        # 过滤文本框
        detection_boxes = self.filter_tag_det_res(detection_boxes, origin_image.shape)
        elapse = time.time() - starttime
        # 返回结果
        return detection_boxes, elapse



# if __name__ == "__main__":
    # args = utility.parse_args()
    # image_file_list = get_image_file_list(args.image_dir)
    # text_detector = TextDetector(args)
    # count = 0
    # total_time = 0
    # draw_img_save = "./output_infer"
    # if not os.path.exists(draw_img_save):
    #     os.makedirs(draw_img_save)
    # for image_file in image_file_list:
    #     img, flag = get_gif_file(image_file)
    #     if not flag:
    #         img = cv2.imread(image_file)
    #     if img is None:
    #         print("error in load img:{}".format(image_file))
    #         continue
    #     dt_boxes, elapse = text_detector(img)
    #     if count > 0:
    #         total_time += elapse
    #     count += 1
    #     print("Predict time is {}: {}".format(image_file, elapse))
    #     src_im = utility.draw_text_det_res(dt_boxes, image_file)
    #     img_name_pure = os.path.split(image_file)[-1]
    #     img_path = os.path.join(draw_img_save,
    #                             "det_res_{}".format(img_name_pure))
    #     cv2.imwrite(img_path, src_im)
    #     print("Image save in {}".format(img_path))
    # if count > 1:
    #     print("Avg Time: {}".format(total_time / (count - 1)))
