import numpy as np
import imgaug
import imgaug.augmenters as iaa


class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        # 参数
        if args is None or len(args) == 0:
            return None
        # 参数是列表
        elif isinstance(args, list):
            if root:
                # 构建序列增强器
                sequence = [self.build(value, root=False) for value in args]
                # 增强
                return iaa.Sequential(sequence)
            else:
                # 返回增强器
                return getattr(iaa, args[0])(*[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            # 获取iaa的增强器类
            cls = getattr(iaa, args['type'])
            # 实例化增强器
            return cls(**{k: self.to_tuple_if_list(v) for k, v in args['args'].items()})
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    # 元组转换
    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment():
    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            # 初始参数
            augmenter_args = [{
                'type': 'Fliplr',
                'args': {
                    'p': 0.5
                }
            }, {
                'type': 'Affine',
                'args': {
                    'rotate': [-10, 10]
                }
            }, {
                'type': 'Resize',
                'args': {
                    'size': [0.5, 3]
                }
            }]
        # 实例化图像增强器
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        # 获取图片数据
        image = data['image']
        shape = image.shape
        # 增强器存在
        if self.augmenter:
            # 图像增强
            aug = self.augmenter.to_deterministic()
            # 使用图像增强器对图像进行增强处理
            data['image'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['polys']:
            # 对多边形进行处理
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        # 转换为np数组
        data['polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        # 将点坐标转换为imgaug.Keypoint
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        # 对点进行数据增强
        keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        # 获取元组列表
        poly = [(p.x, p.y) for p in keypoints]
        return poly
