from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint
    # 评估单张图片的检测结果
    def evaluate_image(self, gt, pred):
        # 并集
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area
        # 交集
        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)
        # 交并比 IoU
        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area
        # 计算平均精读
        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)
                if numGtCare > 0:
                    AP /= numGtCare
            return AP
        
        perSampleMetrics = {}
        matchedSum = 0
        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        arrGlobalConfidences = []
        arrGlobalMatches = []
        recall = 0
        precision = 0
        hmean = 0
        detMatched = 0
        iouMat = np.empty([1, 1])
        gtPols = []
        detPols = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCarePolsNum = []
        detDontCarePolsNum = []
        pairs = []
        detMatchedNums = []
        arrSampleConfidences = []
        arrSampleMatch = []
        evaluationLog = ""

        for n in range(len(gt)):
            # 获取坐标和忽略标签
            points = gt[n]['points']
            dontCare = gt[n]['ignore']
            # 多边形是否有效
            if not Polygon(points).is_valid:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)
        # 记录日志
        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) + " don't care)\n"
            if len(gtDontCarePolsNum) > 0 else "\n")
        # 检测多边形
        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid:
                continue
            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        evaluationLog += "DET polygons: " + str(len(detPols)) + (
            " (" + str(len(detDontCarePolsNum)) + " don't care)\n"
            if len(detDontCarePolsNum) > 0 else "\n")
        # 计算IoU
        if len(gtPols) > 0 and len(detPols) > 0:
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
        # 计算召回率、精确度、调和平均数
        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare
        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        perSampleMetrics = {
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'detMatched': detMatched,
        }
        return perSampleMetrics
    # 合并多张图片的评估结果
    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']
        methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / ( methodRecall + methodPrecision)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }
        return methodMetrics

if __name__ == '__main__':
    evaluator = DetectionIoUEvaluator()
    gts = [[{
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        'text': 5678,
        'ignore': False,
    }]]
    preds = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
