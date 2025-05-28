
from torch import nn
from ocr.modeling.backbones import build_backbone
from ocr.modeling.necks import build_neck
from ocr.modeling.heads import build_head

__all__ = ['BaseModel']


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        # 输入的通道数
        in_channels = config.get('in_channels', 3)
        # 模型的类型
        model_type = config['model_type']
        # 定义主干网络
        if 'Backbone' not in config or config['Backbone'] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            # 设置输入通道数
            config["Backbone"]['in_channels'] = in_channels
            # 构建主干网络
            self.backbone = build_backbone(config["Backbone"], model_type)
            in_channels = self.backbone.out_channels
        # 定义Neck层
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels

        # 定义Head层
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config["Head"]['in_channels'] = in_channels
            self.head = build_head(config["Head"])

        self.return_all_feats = config.get("return_all_feats", False)

    def forward(self, x, data=None):
        y = dict()
        # if self.use_transform:
        #     x = self.transform(x)

        if self.use_backbone:
            x = self.backbone(x)
        
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x

        final_name = "backbone_out"

        if self.use_neck:
            x = self.neck(x)
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            final_name = "neck_out"
        
        if self.use_head:
            x = self.head(x, data=data)
            if 'ctc_neck' in x.keys():
                y["neck_out"] = x["ctc_neck"]
            y["head_out"] = x
            final_name = "head_out"
        if self.return_all_feats:
            if self.training:
                return y
            else:
                return {final_name: x}
        else:
            return x