from argparse import ArgumentParser, RawDescriptionHelpFormatter

import yaml


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        # 添加参数--选择的配置文件
        self.add_argument("-c", "--config")
        # 添加参数--配置额外信息
        self.add_argument(
            "-o", "--opt", nargs='*')

    # 解析命令行参数
    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "无配置文件"
        args.opt = self._parse_opt(args.opt)
        return args

    # 解析其他配置参数
    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config

# 更新识别模型的输出通道数
def update_rec_head_out_channels(cfg, post_process_class):
    if hasattr(post_process_class, 'character'):
        # 获取字符集长度
        char_num = len(getattr(post_process_class, 'character'))
        if cfg['Architecture']['Head']['name'] == 'MultiHead':
            out_channels_list = {}
            # CTC设置
            out_channels_list['CTCLabelDecode'] = char_num
            # 使用SARloss
            if list(cfg['Loss']['loss_config_list'][1].keys())[0] == 'SARLoss':
                if cfg['Loss']['loss_config_list'][1]['SARLoss'] is None:
                    cfg['Loss']['loss_config_list'][1]['SARLoss'] = {'ignore_index': char_num + 1}
                else:
                    cfg['Loss']['loss_config_list'][1]['SARLoss']['ignore_index'] = char_num + 1
                out_channels_list['SARLabelDecode'] = char_num + 2
            cfg['Architecture']['Head']['out_channels_list'] = out_channels_list
        else: 
            cfg['Architecture']["Head"]['out_channels'] = char_num
