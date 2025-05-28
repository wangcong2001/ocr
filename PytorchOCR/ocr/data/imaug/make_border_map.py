import numpy as np
import cv2

np.seterr(divide='ignore', invalid='ignore')
import pyclipper
from shapely.geometry import Polygon
import warnings

warnings.simplefilter("ignore")

__all__ = ['MakeBorderMap']

# 生成文本的边界地图
class MakeBorderMap(object):
    def __init__(self,
                 shrink_ratio=0.4,
                 thresh_min=0.3,
                 thresh_max=0.7,
                 **kwargs):
        # 缩小比例，用于控制生成的边界地图的边界宽度
        self.shrink_ratio = shrink_ratio
        # 边界地图像素值的下阈值
        self.thresh_min = thresh_min
        # 边界地图像素值的上阈值
        self.thresh_max = thresh_max
        # 通过epoch调整shrink_ratio
        if 'total_epoch' in kwargs and 'epoch' in kwargs and kwargs['epoch'] != "None":
            self.shrink_ratio = self.shrink_ratio + 0.2 * kwargs['epoch'] / float(kwargs['total_epoch'])

    def __call__(self, data):
        # 获取数据
        img = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']
        # 创建边界地图
        canvas = np.zeros(img.shape[:2], dtype=np.float32)
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        # 遍历文本框
        for i in range(len(text_polys)):
            if ignore_tags[i]:
                continue
            # 生成边界地图
            self.draw_border_map(text_polys[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        data['threshold_map'] = canvas
        data['threshold_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        # 转换为np数组
        polygon = np.array(polygon)
        # 判断数据是否合法
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2
        # 创建多边形
        polygon_shape = Polygon(polygon)
        # 判断多边形是否合法
        if polygon_shape.area <= 0:
            return
        # 计算边界距离
        distance = polygon_shape.area * ( 1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        # 转换为列表元组
        subject = [tuple(l) for l in polygon]
        # 创建多边形填充操作
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 执行偏移操作，获取偏移后的多边形
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
        # 计算偏移后的多边形宽高
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin
        # 生成网格，计算每个像素点到多边形边界的距离
        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))
        # 创建存储距离地图的数组
        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self._distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        # 距离该像素最近的多边形边界的距离
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height, xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
    
    # 计算点到线的距离
    def _distance(self, xs, ys, point_1, point_2):
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])
        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin /square_distance)
        result[cosin <0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result

    def extend_line(self, point_1, point_2, result, shrink_ratio):
        ex_point_1 = (int(
            round(point_1[0] + (point_1[0] - point_2[0]) * (1 + shrink_ratio))),
            int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + shrink_ratio))))
        cv2.line(result,tuple(ex_point_1),tuple(point_1),4096.0,1,lineType=cv2.LINE_AA,shift=0)
        ex_point_2 = (int(
            round(point_2[0] + (point_2[0] - point_1[0]) * (1 + shrink_ratio))),
            int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + shrink_ratio))))
        cv2.line(result,tuple(ex_point_2),tuple(point_2),4096.0,1,lineType=cv2.LINE_AA,shift=0)
        return ex_point_1, ex_point_2
