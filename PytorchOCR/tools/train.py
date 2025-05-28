import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.utility import ArgsParser
from ocr import Config
from ocr import Trainer


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        default=True,
        help="Whether to perform evaluation in train")
    args = parser.parse_args()
    return args


def main():
    # 初始化参数配置
    FLAGS = parse_args()
    # 获取配置文件路径
    cfg = Config(FLAGS.config)
    # 将配置文件转换成字典
    FLAGS = vars(FLAGS)
    # 获取opt参数
    opt = FLAGS.pop('opt')
    # 合并参数字典
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    # 启动训练实例
    trainer = Trainer(cfg, mode='train_eval' if FLAGS['eval'] else 'train')
    trainer.train()


if __name__ == '__main__':
    main()
