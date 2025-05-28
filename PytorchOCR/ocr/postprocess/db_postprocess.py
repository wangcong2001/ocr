import numpy as np
import cv2
import torch
from shapely.geometry import Polygon
import pyclipper

# 从二值化图像中提取文本框并根据置信度阈值进行过滤
class DBPostProcess(object):
    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 box_type='quad',
                 **kwargs):
        # 二值化图像的阈值
        self.thresh = thresh
        # 文本框置信度
        self.box_thresh = box_thresh
        # 候选文本框的最大数量
        self.max_candidates = max_candidates
        # 文本框裁剪比例
        self.unclip_ratio = unclip_ratio
        # 最小尺寸
        self.min_size = 3
        # 文本框的计分模式
        self.score_mode = score_mode
        # 文本框类型
        self.box_type = box_type
        assert score_mode in ["slow", "fast" ], "后处理错误"
        # 设置膨胀核
        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

        
    # 二值图像中提取多边形
    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        # 获取二值图像
        bitmap = _bitmap
        height, width = bitmap.shape

        boxes = []
        scores = []
        # 查找所有轮廓
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 遍历所有轮廓知道最大候选数量
        for contour in contours[:self.max_candidates]:
            # 轮廓的近似精度
            epsilon = 0.002 * cv2.arcLength(contour, True)
            # 进行多边形逼近
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # 多边形的顶点集
            points = approx.reshape((-1, 2))
            # 顶点集数量不足
            if points.shape[0] < 4:
                continue
            # 计算分数
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            # 置信度不足
            if self.box_thresh > score:
                continue
            # 点数量大于2
            if points.shape[0] > 2:
                # 解卷
                box = self.unclip(points, self.unclip_ratio)
                # 文本框分裂
                if len(box) > 1:
                    continue
            else:
                continue
            # 转换为二维数组
            box = box.reshape(-1, 2)
            # 获取最小矩形和边长
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            # 边长过小
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            # 限定宽度
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            # 限定高度
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            # 添加文本框
            boxes.append(box.tolist())
            # 添加分数
            scores.append(score)
        return boxes, scores
    
    # 二值图像中提取矩形
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap
        height, width = bitmap.shape
        # 获取轮廓
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 获取信息
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        # 限定提取数量
        num_contours = min(len(contours), self.max_candidates)
        boxes = []  
        scores = []
        for index in range(num_contours):
            # 获取contour
            contour = contours[index]
            # 获取最小矩形和边长
            points, sside = self.get_mini_boxes(contour)
            # 边长国小
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))

            # 计算得分
            # if self.score_mode == "fast":
            #     score = self.box_score_fast(pred, points.reshape(-1, 2))
            # else:
            #     score = self.box_score_slow(pred, contour)


            if self.box_thresh > score:
                continue
            # 解卷
            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            # 获取最小矩形和边长
            box, sside = self.get_mini_boxes(box)
            # 边长过小
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    # 对多边形的解卷
    def unclip(self, box, unclip_ratio):
        # 实例化多边形
        poly = Polygon(box)
        # 计算距离
        distance = poly.area * unclip_ratio / poly.length
        # 创建多边形偏移
        offset = pyclipper.PyclipperOffset()
        # 添加路径到执行操作之中
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 执行偏移
        expanded = np.array(offset.Execute(distance))
        return expanded
    # 获取文本框的最小矩形
    def get_mini_boxes(self, contour):
        # 获取包围轮廓的最小边界框
        bounding_box = cv2.minAreaRect(contour)
        # 获取最小边界框的四个顶点坐标
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        # 确定四个顶点的顺序 逆时针排序
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
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
        # 获取四个顶点
        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        # 返回最小矩形的四个顶点坐标
        return box, min(bounding_box[1])

    # 计算文本框的得分
    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        # 获取文本框的坐标
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)
        # 创建0数组
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        # 对文本框进行偏移
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        # 图像填充
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        # 返回均值
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    # 计算分数
    # def box_score_slow(self, bitmap, contour):
    #     h, w = bitmap.shape[:2]
    #     contour = contour.copy()
    #     contour = np.reshape(contour, (-1, 2))

    #     xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
    #     xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
    #     ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
    #     ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

    #     mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

    #     contour[:, 0] = contour[:, 0] - xmin
    #     contour[:, 1] = contour[:, 1] - ymin

    #     cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
    #     return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, preds, batch):
        # 获取预测结果
        preds = preds['res']
        # 样本的形状信息
        shape_list = batch[1]
        # 将tensor转化为数组
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(shape_list, torch.Tensor):
            shape_list = shape_list.cpu().numpy()
        # 获取第一个通道的结果
        pred = preds[:, 0, :, :]
        # 二值化图像
        segmentation = pred > self.thresh
        # 文本框列表
        boxes_batch = []

        for batch_index in range(pred.shape[0]):
            # 获取样本的形状信息
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            # 膨胀核
            if self.dilation_kernel is not None:
                mask = cv2.dilate( np.array(segmentation[batch_index]).astype(np.uint8), self.dilation_kernel)
            else:
                # 获取二值化图像
                mask = segmentation[batch_index]
            # 文本框类型
            if self.box_type == 'poly':
                # 多边形
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], mask, src_w, src_h)
            elif self.box_type == 'quad':
                # 矩形
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            else:
                raise ValueError("后处理错误")
            # 添加文本框
            boxes_batch.append({'points': boxes})
        return boxes_batch
