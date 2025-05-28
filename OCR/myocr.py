import time
import cv2
from OCR.tools.infer.predict_infer import TextSystem
from OCR.pytorchocr.utils.img_util import get_image_file_list, get_gif_file
import os
import OCR.tools.infer.predict_config as args

__dir__ = os.path.dirname(os.path.abspath(__file__))


class PytorchOcr:
    def __init__(self):
        self.text_sys = TextSystem(args)

    def predict(self, img_path):
        args.image_dir = img_path
        drop_score = args.drop_score
        image_file_list = get_image_file_list(img_path)
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
            detection_boxs, rec_res = self.text_sys(img)
            elapse = time.time() - starttime
            # print("Predict time is %s: %.3fs" % (image_file, elapse))
            ret = []
            for idx in range(len(detection_boxs)):
                box = [tuple(x) for x in detection_boxs[idx]]
                text = rec_res[idx][0]
                score = rec_res[idx][1]
                item = {
                    "box": box,
                    "text": text,
                    "score": score
                }
                print("box: ", box)
                print("line: ", text)
                print("score: ", score)
                ret.append(item)
            return ret

#
# pytorchocr = PytorchOcr()
# img_path = os.path.join(__dir__, './input/page-18_2024_03_07_20_26_26.png')
# ret = pytorchocr.predict(img_path)
# print(ret)
