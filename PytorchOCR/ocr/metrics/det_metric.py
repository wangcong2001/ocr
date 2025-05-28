__all__ = ['DetMetric']

from .eval_det_iou import DetectionIoUEvaluator

# 计算目标检测模型的精确度、召回率和 F1 分数
class DetMetric(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        # 目标检测的评估器
        self.evaluator = DetectionIoUEvaluator()
        # 指定主要的评估指标，默认为 F1 分数
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        # 获取标签内的多边形坐标
        gt_polyons_batch = batch[2].cpu().numpy()
        ignore_tags_batch = batch[3].cpu().numpy()
        # 遍历预测结果
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch, ignore_tags_batch):
            # 信息列表
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # 检测结果的信息列表
            det_info_list = [{
                'points': det_polyon,
                'text': ''
            } for det_polyon in pred['points']]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)

    def get_metric(self):
        metrics = self.evaluator.combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = [] 

