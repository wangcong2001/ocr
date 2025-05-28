import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

__all__ = ['MakeShrinkMap']


class MakeShrinkMap(object):

    def __init__(self, min_text_size=8, shrink_ratio=0.4, **kwargs):
        # 设置文本的最小尺寸
        self.min_text_size = min_text_size
        # 控制生成的缩小地图的缩小比例
        self.shrink_ratio = shrink_ratio
        if 'total_epoch' in kwargs and 'epoch' in kwargs and kwargs['epoch'] != "None":
            self.shrink_ratio = self.shrink_ratio + 0.2 * kwargs['epoch'] / float(kwargs['total_epoch'])

    def __call__(self, data):
        # 获取数据
        image = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        # 验证文本框是否合法
        text_polys, ignore_tags = self.validate_polygons(text_polys,ignore_tags, h, w)
        # 创建0数据
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)

        for i in range(len(text_polys)):
            # 获取文本框
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])

            # 是否忽略
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask,polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                # 实例化多边形
                polygon_shape = Polygon(polygon)
                subject = [tuple(l) for l in polygon]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,pyclipper.ET_CLOSEDPOLYGON)
                shrinked = []

                # 调整收缩比例
                possible_ratios = np.arange(self.shrink_ratio, 1,self.shrink_ratio)
                np.append(possible_ratios, 1)
                # 计算收缩后的多边形
                for ratio in possible_ratios:
                    distance = polygon_shape.area * (1 - np.power(ratio, 2)) / polygon_shape.length

                    shrinked = padding.Execute(-distance)
                    if len(shrinked) == 1:
                        break
                # 忽略
                if shrinked == []:
                    cv2.fillPoly(mask,polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                # 成功收缩后
                for each_shirnk in shrinked:
                    # np数组
                    shirnk = np.array(each_shirnk).reshape(-1, 2)
                    # 填充区域
                    cv2.fillPoly(gt, [shirnk.astype(np.int32)], 1)
        # 数据更新
        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        # 为空直接返回
        if len(polygons) == 0:
            return polygons, ignore_tags
        # 数量不相同
        assert len(polygons) == len(ignore_tags)
        # 遍历文本框
        for polygon in polygons:
            # 限制文本框的坐标
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            # 计算面积
            area = self.polygon_area(polygons[i])
            # 面积过小直接忽略
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        area = 0
        q = polygon[-1]
        for p in polygon:
            area += p[0] * q[1] - p[1] * q[0]
            q = p
        return area / 2.0
