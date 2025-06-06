import numpy as np

# 图像变换类
class WarpMLS:
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    # 双线性插值函数
    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 *(1 - y) + v22 * y) * x

    # 生成图像
    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        # 初始化
        w = np.zeros(self.pt_count, dtype=np.float32)
        
        if self.pt_count < 2:
            return

        i = 0
        while 1:
            # 确保i在图像内
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break
            j = 0
            while 1:
                # 确保j在图像内
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break
                # 权重之和
                sw = 0
                # 权重乘以目标点坐标的累加和
                swp = np.zeros(2, dtype=np.float32)
                # 权重乘以源点坐标的累加和
                swq = np.zeros(2, dtype=np.float32)
                # 计算得到的新点的坐标
                new_pt = np.zeros(2, dtype=np.float32)
                # 当前点的坐标
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break
                    # 计算权重
                    w[k] = 1. / (
                        (i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0]) +
                        (j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1]))
                    # 累加
                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])
                if k == self.pt_count - 1:
                    # 计算均值点
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq
                    # 累计权重
                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        # 目标点距离均值点的偏移量
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)
                    # 当前点的偏移量
                    cur_pt -= pstar
                    # 旋转
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        # 当前点忽略
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        # 目标点距离均值点的偏移量
                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])
                        tmp_pt = np.zeros(2, dtype=np.float32)
                        # 计算新的坐标
                        tmp_pt[0] = np.sum(pt_i * cur_pt) * self.src_pts[k][0] - np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        # 新坐标乘以权重
                        tmp_pt *= (w[k] / miu_s)
                        # 添加进新坐标
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]
                # 仿射变换后的新位置
                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    # 生成图像
    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                # 确定网格的边界
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1
                # 计算像素的偏移量
                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp( di / h, dj / w, self.rdx[i, j], self.rdx[i, nj],self.rdx[ni, j], self.rdx[ni, nj])
                delta_y = self.__bilinear_interp(di / h, dj / w, self.rdy[i, j], self.rdy[i, nj],self.rdy[ni, j], self.rdy[ni, nj])

                # 计算变换后的坐标
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                # 限制坐标边界
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)
                
                if len(self.src.shape) == 3:
                    # 扩展通道
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                # 双线性插值
                dst[i:i + h, j:j + w] = self.__bilinear_interp(x, y, self.src[nyi, nxi], self.src[nyi, nxi1],self.src[nyi1, nxi], self.src[nyi1, nxi1])
        # 限制边界
        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst
