import torch
import torch.nn as nn
import torch.nn.functional as F


class SAREncoder(nn.Module):
    def __init__(self,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.0,
                 enc_gru=False, # GRU 或 LSTM
                 d_model=512, # 输入特征的通道数
                 d_enc=512,  # 编码器的RNN层隐藏状态维度
                 mask=True,
                 **kwargs):
        super().__init__()

        self.enc_bi_rnn = enc_bi_rnn
        self.enc_drop_rnn = enc_drop_rnn
        self.mask = mask
        # 构建RNN的参数
        kwargs = dict(
            input_size=d_model, # 输入特征大小
            hidden_size=d_enc, # 隐藏状态大小
            num_layers=2, # RNN层数
            batch_first=True, # 输入输出的第一个维度是batch
            dropout=enc_drop_rnn, # 丢弃率
            bidirectional=enc_bi_rnn # 是否双向
            ) 
        
        # 构建RNN
        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)

        # rnn的输出大小 双向x2
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        # 线性层
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    # 前馈
    def forward(self, feat, img_metas=None):
        # 是否存在元信息
        if img_metas is not None:
            assert len(img_metas[0]) == feat.size(0)
        valid_ratios = None
        # 是否需要掩码
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]
        # b c h w 获取高度特征
        h_feat = feat.size(2)  
        # 最大池化
        feat_v = F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        # 去除维度为1的维度 b c w
        feat_v = feat_v.squeeze(2)
        # 转换维度 b w c
        feat_v = feat_v.permute(0, 2, 1).contiguous()
        # rnn编码 b * t * c
        holistic_feat = self.rnn_encoder(feat_v)[0]  

        if valid_ratios is not None:
            # 获取有效的特征
            valid_hf = []
            # 获取特征的时间步
            T = holistic_feat.size(1)
            # 根据比例值计算有效步数
            for i in range(valid_ratios.size(0)):
                valid_step = torch.minimum(torch.tensor(T), torch.ceil(T * valid_ratios[i]).int()) - 1
                # 仅截取有效步数
                valid_hf.append(holistic_feat[i, valid_step, :])
            # 拼接
            valid_hf = torch.stack(valid_hf, dim=0)
        else:
            # 没有元图像则取最后一个时间步
            valid_hf = holistic_feat[:, -1, :]  
        # 线性层
        holistic_feat = self.linear(valid_hf) 

        return holistic_feat


class BaseDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                label=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode

        if train_mode:
            return self.forward_train(feat, out_enc, label, img_metas)
        return self.forward_test(feat, out_enc, img_metas)


class ParallelSARDecoder(BaseDecoder):
    def __init__(
            self,
            out_channels,  # 输出通道数
            enc_bi_rnn=False, # 是否使用双向 RNN
            dec_bi_rnn=False, 
            dec_drop_rnn=0.0,
            dec_gru=False,
            d_model=512, # 输入特征的通道数
            d_enc=512, # 编码器的RNN层隐藏状态维度
            d_k=64,     # 注意力的维度或特征维度
            pred_dropout=0.0,
            max_text_length=30,
            mask=True,
            pred_concat=True,
            **kwargs):
        super().__init__()

        self.num_classes = out_channels
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = out_channels - 2 # 开始索引
        self.padding_idx = out_channels - 1 # 填充索引
        self.max_seq_len = max_text_length 
        self.mask = mask
        self.pred_concat = pred_concat
        # 编码器的RNN层输出大小
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        # 解码器的RNN层输出大小
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)

        # 线性层
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        # 卷积层
        self.conv3x3_1 = nn.Conv2d(d_model, d_k, kernel_size=3, stride=1, padding=1)
        # 线性层
        self.conv1x1_2 = nn.Linear(d_k, 1)

        kwargs = dict(
            input_size=encoder_rnn_out_size,
            hidden_size=encoder_rnn_out_size,
            num_layers=2,
            batch_first=True,
            dropout=dec_drop_rnn,
            bidirectional=dec_bi_rnn)
        # 解码器的RNN
        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)

        # 数据嵌入层
        self.embedding = nn.Embedding(
            self.num_classes,
            encoder_rnn_out_size,
            padding_idx=self.padding_idx)

        # 预测的丢弃操作
        self.pred_dropout = nn.Dropout(pred_dropout)

        pred_num_classes = self.num_classes - 1

        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + encoder_rnn_out_size
        else:
            fc_in_channel = d_model
        # 线性层
        self.prediction = nn.Linear(fc_in_channel, pred_num_classes)

    # 注意力机制
    def _2d_attention(self,
                      decoder_input,
                      feat,
                      holistic_feat,
                      valid_ratios=None):

        # 解码器的RNN b (seq_len + 1) hidden_size
        y = self.rnn_decoder(decoder_input)[0]
        # 线性层 b (seq_len + 1) attn_size
        attn_query = self.conv1x1_1(y)  # 
        # 获取形状
        bsz, seq_len, attn_size = attn_query.shape
        # 转换维度 b, seq_len, attn_size, 1, 1
        attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)
        # 卷积层 b, attn_size, h, w
        attn_key = self.conv3x3_1(feat)
        # 添加维度 b, 1, attn_size, h, w
        attn_key = attn_key.unsqueeze(1)
        # 计算注意力权重
        attn_weight = torch.tanh(torch.add(attn_key, attn_query))
        # 转换维度 
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        # 线性层
        attn_weight = self.conv1x1_2(attn_weight)
        # 获取维度
        bsz, T, h, w, c = attn_weight.size()
        assert c == 1

        if valid_ratios is not None:
            # 对注意力权重进行掩码操作
            for i in range(valid_ratios.size(0)):
                valid_width = torch.minimum(torch.tensor(w), torch.ceil(w * valid_ratios[i]).int())
                if valid_width < w:
                    attn_weight[i, :, :, valid_width:, :] = float('-inf')
        # 转换维度大小
        attn_weight = attn_weight.view(bsz, T, -1)
        # 计算注意力权重
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.view(bsz, T, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
        attn_feat = torch.sum(torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False)

        # 预测
        if self.pred_concat:
            # 全局特征的通道数或特征维度的大小
            hf_c = holistic_feat.shape[-1]
            # 扩展维度
            holistic_feat = holistic_feat.expand(bsz, seq_len, hf_c)
            # 拼接预测
            y = self.prediction(torch.cat((y, attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(attn_feat)
        # 训练模式下进行丢弃操作
        if self.train_mode:
            y = self.pred_dropout(y)
        return y

    def forward_train(self, feat, out_enc, label, img_metas):
        # 获取元信息
        if img_metas is not None:
            assert img_metas[0].size(0) == feat.size(0)
        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]

        # 标签的嵌入
        lab_embedding = self.embedding(label)
        # 扩展维度
        out_enc = out_enc.unsqueeze(1)
        # 拼接
        in_dec = torch.cat((out_enc, lab_embedding), dim=1)
        # 注意力机制
        out_dec = self._2d_attention(in_dec, feat, out_enc, valid_ratios=valid_ratios)
        # 输出进行切片，去除起始标记
        return out_dec[:, 1:, :]

    def forward_test(self, feat, out_enc, img_metas):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]
        # 序列的最大长度
        seq_len = self.max_seq_len
        # 批量大小
        bsz = feat.size(0)
        # 创建了起始标记
        start_token = torch.full((bsz, ), fill_value=self.start_idx, device=feat.device,dtype=torch.long)
        # 嵌入
        start_token = self.embedding(start_token)
        # 获取嵌入向量的维度
        emb_dim = start_token.shape[1]
        # 扩展维度
        start_token = start_token.unsqueeze(1).expand(bsz, seq_len, emb_dim)
        out_enc = out_enc.unsqueeze(1)
        # 拼接
        decoder_input = torch.cat((out_enc, start_token), dim=1)
        outputs = []
        # 生成输出序列
        for i in range(1, seq_len + 1):
            # 计算输出
            decoder_output = self._2d_attention(decoder_input, feat, out_enc, valid_ratios=valid_ratios)
            # 选择当前时间步的输出
            char_output = decoder_output[:, i, :] 
            # 转化为概率
            char_output = F.softmax(char_output, -1)
            # 添加到输出列表
            outputs.append(char_output)
            # 获取最大值的索引
            _, max_idx = torch.max(char_output, dim=1, keepdim=False)
            # 在嵌入矩阵检索最大概率
            char_embedding = self.embedding(max_idx)
            # 作为下一个输入
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding
        # 输出堆叠
        outputs = torch.stack(outputs, 1) 

        return outputs


class SARHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 enc_dim=512, # 编码器的隐藏层维度
                 max_text_length=30, # 文本序列的最大长度
                 enc_bi_rnn=False, # 是否使用双向循环神经网络
                 enc_drop_rnn=0.1, # 循环神经网络的丢弃率
                 enc_gru=False, # 是否使用门控循环单元（GRU）
                 dec_bi_rnn=False, # 是否使用双向循环神经网络
                 dec_drop_rnn=0.0, # 循环神经网络的丢弃率
                 dec_gru=False, # 是否使用GRU
                 d_k=512,  # 注意力机制中的键和查询的维度
                 pred_dropout=0.1, # 预测的丢弃率
                 pred_concat=True, # 预测阶段进行连接操作
                 **kwargs):
        super(SARHead, self).__init__()

        # 编码模块
        self.encoder = SAREncoder(
            enc_bi_rnn=enc_bi_rnn,
            enc_drop_rnn=enc_drop_rnn,
            enc_gru=enc_gru,
            d_model=in_channels,
            d_enc=enc_dim)

        # 解码模块
        self.decoder = ParallelSARDecoder(
            out_channels=out_channels,
            enc_bi_rnn=enc_bi_rnn,
            dec_bi_rnn=dec_bi_rnn,
            dec_drop_rnn=dec_drop_rnn,
            dec_gru=dec_gru,
            d_model=in_channels,
            d_enc=enc_dim,
            d_k=d_k,
            pred_dropout=pred_dropout,
            max_text_length=max_text_length,
            pred_concat=pred_concat)

    def forward(self, feat, data=None):
        holistic_feat = self.encoder(feat, data)  # bsz c
        if self.training:
            label = data[0]  # label
            final_out = self.decoder(
                feat, holistic_feat, label, img_metas=data)
        else:
            final_out = self.decoder(
                feat,
                holistic_feat,
                label=None,
                img_metas=data,
                train_mode=False)
            # (bsz, seq_len, num_classes)

        return {'res': final_out}
