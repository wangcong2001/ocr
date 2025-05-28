import numpy as np
import cv2
import torch
from shapely.geometry import Polygon
import pyclipper

# 后处理
class DBPostProcess(object):

    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, 
                 unclip_ratio=2.0, **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    # 从位图中获取文本框
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):

        bitmap = _bitmap
        height, width = bitmap.shape

        # 使用cv寻找二值化轮廓
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), 
                                cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取轮廓和层次
        contours, _ = outs[0], outs[1]
        # 获取轮廓数量
        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        # 遍历轮廓
        for contours_idx in range(num_contours):
            contour = contours[contours_idx]

            # 获取最小外接矩形和最小边长
            points, short_side = self.get_mini_boxes(contour)
            # 如果最小边长小于最小边长阈值，跳过
            if short_side < self.min_size:
                continue
            points = np.array(points)
            # 计算文本框的得分
            score = self.box_score(pred, contour)
            # 如果得分小于文本框得分阈值，跳过
            if self.box_thresh > score:
                continue
            # 获取多边形文本框
            box = self.unclip(points).reshape(-1, 1, 2)
            # 获取文本框最小外接矩形和最小边长
            box, short_side = self.get_mini_boxes(box)
            # 如果最小边长小于最小边长阈值，跳过
            if short_side < self.min_size + 2:
                continue
            box = np.array(box)
            # 调整文本框大小 转换空间坐标
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            # 添加列表
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        # 传入参数比例
        unclip_ratio = self.unclip_ratio
        # 实例化多边形对象
        poly = Polygon(box)
        # 向外扩展的距离
        distance = poly.area * unclip_ratio / poly.length
        # 偏移操作获取封闭多边形
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 执行偏移操作并获取文本框多边形
        expanded = np.array(offset.Execute(distance))
        return expanded

    # 获取轮廓最小坐标
    def get_mini_boxes(self, contour):
        # 计算最小外接矩形
        bounding_box = cv2.minAreaRect(contour)
        # 获取最小外接矩形的四个顶点坐标并排序
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        # 左上、左下、右下、右上
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        # 左上、左下、右下、右上
        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        # 返回坐标和最小边长
        return box, min(bounding_box[1])

    def box_score(self, bitmap, contour):
        h, w = bitmap.shape[:2]
        # 复制轮廓
        contour = contour.copy()
        # 调整维数
        contour = np.reshape(contour, (-1, 2))
        # 计算轮廓的外接矩形
        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)
        # 掩码轮廓内部置1
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        # 平移变换
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin
        # 填充轮廓
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        # 返回轮廓内均值
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        # 获取预测结果
        pred = outs_dict['maps']
        # 转化为数组
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        # bchw 获取切片
        pred = pred[:, 0, :, :]
        # 二值化
        segmentation = pred > self.thresh

        boxes_batch = []
        # 遍历批次
        for batch_index in range(pred.shape[0]):
            src_heigh, src_width, _, _ = shape_list[batch_index]
            # 获取二至分割的掩码
            mask = segmentation[batch_index]
            # 调用函数获取文本框和得分
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_width, src_heigh)

            boxes_batch.append({'points': boxes, 'scores': scores})
        return boxes_batch