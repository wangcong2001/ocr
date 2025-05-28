import os
import sys
import cv2
import copy
import numpy as np
import time
import OCR.tools.infer.pytorchocr_utility as utility
import OCR.tools.infer.predict_config as args
import OCR.tools.infer.predict_recognization as predict_rec
import OCR.tools.infer.predict_detection as predict_det
from OCR.pytorchocr.utils.img_util import get_image_file_list, get_gif_file

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))


class TextSystem(object):
    def __init__(self, args, **kwargs):
        # 图像分割
        self.text_detector = predict_det.TextDetector(args, **kwargs)
        # 图像识别
        self.text_recognizer = predict_rec.TextRecognizer(args, **kwargs)

        self.drop_score = args.drop_score

    # 获取一个以文本框为基础旋转裁剪后的图像
    def get_rotate_crop_image(self, img, points):
        # 计算最大宽度
        img_crop_width = int(
            max(
                # 计算欧式距离
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        # 计算最大高度
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        # 表示四个脚坐标
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        #  获取透视变换矩阵 将点映过去
        M = cv2.getPerspectiveTransform(points, pts_std)
        # 进行透视变换
        dest_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,  # 复制边界
            flags=cv2.INTER_CUBIC)  # 双三次插值

        dest_img_height, dest_img_width = dest_img.shape[0:2]
        # 如果高度大于宽度，进行旋转
        if dest_img_height * 1.0 / dest_img_width >= 1.5:
            dest_img = np.rot90(dest_img)
        return dest_img

    def __call__(self, img):
        origin_image = img.copy()
        # 分割图片
        detection_boxs, elapse = self.text_detector(img)
        # print("detection_boxs num : {}, elapse : {}".format(
        #     len(detection_boxs), elapse))

        # 没有检测到文本框
        if detection_boxs is None:
            return None, None

        img_crop_list = []

        # 排序确保文本框按照从上到下、从左到右的顺序排列
        detection_boxs = sorted_boxes(detection_boxs)

        # 识别文本框
        for box_idx in range(len(detection_boxs)):
            tmp_box = copy.deepcopy(detection_boxs[box_idx])
            img_crop = self.get_rotate_crop_image(origin_image, tmp_box)
            img_crop_list.append(img_crop)
        # 识别文本
        rec_res, elapse = self.text_recognizer(img_crop_list)
        print("text recognizer num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # 过滤低置信度文本框
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(detection_boxs, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        # 返回文本框和识别结果
        return filter_boxes, filter_rec_res


# 排序确保文本框按照从上到下、从左到右的顺序排列
def sorted_boxes(detection_boxs):
    num_boxes = detection_boxs.shape[0]
    sorted_boxes = sorted(detection_boxs, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main():
    # 获取图像列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建实例
    text_sys = TextSystem(args)
    # 低置信度过滤阈值
    drop_score = args.drop_score
    # 遍历图像列表
    for image_file in image_file_list:
        # 判断是否为gif图像
        img, flag = get_gif_file(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in load img:{}".format(image_file))
            continue
        # 获取信息
        starttime = time.time()
        detection_boxs, rec_res = text_sys(img)
        elapse = time.time() - starttime
        # print("Predict time is %s: %.3fs" % (image_file, elapse))

        for text, score in rec_res:
            print("{}, {:.3f}".format(text, score))

# if __name__ == '__main__':
#     args.image_dir = os.path.join(os.path.abspath(os.path.join(__dir__, '../..')), './input/page-18_2024_03_07_20_26_26.png')
#     main()
