import os
import sys

# 获取当前目录
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录加入环境变量
sys.path.append(__dir__)
# 将当前目录的上一级目录加入环境变量
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
import cv2
import numpy as np
import math
import time
import torch
from OCR.tools.infer.base import Base
import OCR.tools.infer.pytorchocr_utility as utility
from OCR.pytorchocr.postprocess import build_post_process
# from pytorchocr.utils.img_util import get_image_file_list, get_gif_file


class TextRecognizer(Base):
    def __init__(self, args, **kwargs):
        # 参数解析
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.character_type = args.rec_char_type
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        self.max_text_length = args.max_text_length
        # 使用CRNN模型(虽然只有这一个)
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": args.rec_char_type,
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
            }
        self.postprocess_op = build_post_process(postprocess_params)

        # cuda加速
        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu

        # 获取参数
        self.limited_max_width = args.limited_max_width
        self.limited_min_width = args.limited_min_width
        self.weights_path = args.rec_model_path
        # 加载数据
        network_config = utility.LoadConfig(self.weights_path)

        # 读取权重
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError('{} is not exist.'.format(self.weights_path))
        weights = torch.load(self.weights_path)
        # 设定输出通道数
        self.out_channels = self.get_out_channels(weights)
        # 修改参数
        kwargs['out_channels'] = self.out_channels
        # 构建网络
        super(TextRecognizer, self).__init__(network_config, **kwargs)
        # 加载权重
        self.load_state_dict(weights)
        # 设置推理模式（评估）
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()

    # 数据缩放和归一化
    def resize_norm_img(self, img, max_wh_ratio):
        # 模型期望的图像尺寸
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        # 计算最大长宽比例
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        # 获取最大值
        imgW = int((imgH * max_wh_ratio))
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        #  获取原始图像的高度和宽度
        h, w = img.shape[:2]
        # 计算缩放比例
        ratio = w / float(h)
        # 计算缩放后的宽度
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)
        # 按照比例缩放
        if ratio_imgH > imgW:
            resized_w = imgW
        else:
            resized_w = int(ratio_imgH)
        # 缩放
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        # 归一化
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        # 填充0
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    
    def __call__(self, img_list):
        # 获取图像数量
        img_num = len(img_list)
        width_list = []
        # 获取所有图像的宽高比例
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # 对宽高比例进行排序并获取索引值
        indices = np.argsort(np.array(width_list))
        # 初始化二维列表
        rec_res = [['', 0.0]] * img_num
        # 获取批次数量
        batch_num = self.rec_batch_num
        # 初始化耗时
        elapse = 0
        # 对每个批次进行处理
        for beg_img_idx in range(0, img_num, batch_num):
            # 获取批次结束的索引
            end_img_idx = min(img_num, beg_img_idx + batch_num)
            # 初始化
            norm_img_batch = []
            max_wh_ratio = 0
            # 获取max_wh_ratio值
            for idx in range(beg_img_idx, end_img_idx):
                # 获取图像的高度和宽度
                h, w = img_list[indices[idx]].shape[0:2]
                # 计算最大宽高比例
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            # 对每个批次中的图像进行处理
            for idx in range(beg_img_idx, end_img_idx):
                norm_img = self.resize_norm_img(img_list[indices[idx]],
                                                max_wh_ratio)
                # 添加维度
                norm_img = norm_img[np.newaxis, :]
                # 添加到列表中
                norm_img_batch.append(norm_img)

            # 将列表内的图像链接成一个批次
            norm_img_batch = np.concatenate(norm_img_batch)
            # 复制
            norm_img_batch = norm_img_batch.copy()
            # 记录时间
            starttime = time.time()
            # 推理
            with torch.no_grad():
                inp = torch.from_numpy(norm_img_batch)
                if self.use_gpu:
                    inp = inp.cuda()
                prob_out = self.net(inp)
            # 检测是否是列表并转换位nunpy数组
            if isinstance(prob_out, list):
                preds = [v.cpu().numpy() for v in prob_out]
            else:
                preds = prob_out.cpu().numpy()
            # 后处理
            rec_result = self.postprocess_op(preds)
            # 保存结果（文字+置信度）按照索引顺序
            for idx in range(len(rec_result)):
                rec_res[indices[beg_img_idx + idx]] = rec_result[idx]
            # 记录结束时间
            elapse += time.time() - starttime
        return rec_res, elapse


# def main(args):
#     image_file_list = get_image_file_list(args.image_dir)
#     text_recognizer = TextRecognizer(args)
#     valid_image_file_list = []
#     img_list = []
#     for image_file in image_file_list:
#         img, flag = get_gif_file(image_file)
#         if not flag:
#             img = cv2.imread(image_file)
#         if img is None:
#             print("error in load image:{}".format(image_file))
#             continue
#         valid_image_file_list.append(image_file)
#         img_list.append(img)
#     rec_res, predict_time = text_recognizer(img_list)
#     try:
#         rec_res, predict_time = text_recognizer(img_list)
#     except Exception as e:
#         print(e)
#         exit()
#     for idx in range(len(img_list)):
#         print("Predicts of {}:{}".format(valid_image_file_list[idx], rec_res[
#             idx]))
#     print("Total predict time for {} images, cost: {:.3f}".format(
#         len(img_list), predict_time))

# if __name__ == '__main__':
#     main(utility.parse_args())