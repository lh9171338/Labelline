import numpy as np
import cv2


class Fisheye():
    def __init__(self, eps=1e-10):
        self.eps = eps

    def interp_line(self, lines, num=None, resolution=0.1):
        lines = lines.copy()
        pts_list = []
        for line in lines:
            # 计算均值
            means = line.mean(axis=0)[None]
            line -= means

            # 计算线段点的主方向
            _, _, R = np.linalg.svd(line)

            # 旋转线段点，使点尽量沿x轴分布
            line = np.matmul(line, R.T)

            # 估计圆锥曲线参数
            n_pts = len(line)
            if n_pts < 5:
                # 一次曲线方程：2 D x + 2 E y + F = 0
                # 矩阵形式 M x = 0
                M = np.zeros((n_pts, 3))
                for i in range(n_pts):
                    x, y = line[i, 0], line[i, 1]
                    M[i, 0] = 2 * x
                    M[i, 1] = 2 * y
                    M[i, 2] = 1
                # 奇异值分解计算零空间
                u, s, vt = np.linalg.svd(M)
                A, B, C = 0.0, 0.0, 0.0
                D, E, F = vt[-1]
            else:
                # 2次曲线方程: A x^2 + 2 B x y + C y^2 + 2 D x + 2 E y + F = 0
                # 矩阵形式 M x = 0
                M = np.zeros((n_pts, 6))
                for i in range(n_pts):
                    x, y = line[i, 0], line[i, 1]
                    M[i, 0] = x ** 2
                    M[i, 1] = 2 * x * y
                    M[i, 2] = y ** 2
                    M[i, 3] = 2 * x
                    M[i, 4] = 2 * y
                    M[i, 5] = 1
                # 奇异值分解计算零空间
                _, _, vt = np.linalg.svd(M)
                A, B, C, D, E, F = vt[-1]

            # 2次曲线插值
            xs0, ys0 = line[:, 0], line[:, 1]
            xmin = np.min(xs0)
            xmax = np.max(xs0)
            dx = xmax - xmin

            K = int(round(dx / resolution)) + 1 if num is None else num
            xs = np.linspace(xmin, xmax, K)

            # 求解：A x^2 + 2 B x y + C y^2 + 2 D x + 2 E y + F = 0
            a = C
            b = 2 * B * xs + 2 * E
            c = A * xs * xs + 2 * D * xs + F
            if abs(a) < self.eps:
                # 解一元一次方程：b y + c = 0
                ys = -1.0 * c / b
            else:
                # 解一元二次方程: a y^2 + b y + c = 0
                delta2 = b * b - 4 * a * c
                delta2[delta2 < 0] = 0
                delta = np.sqrt(delta2)

                x0, y0 = xs0[n_pts // 2], ys0[n_pts // 2]
                b0 = 2 * B * x0 + 2 * E
                c0 = A * x0 * x0 + 2 * D * x0 + F
                delta0 = np.sqrt(b0 * b0 - 4 * a * c0)
                y0_1 = 0.5 * (-1.0 * b0 + delta0) / a
                y0_2 = 0.5 * (-1.0 * b0 - delta0) / a
                if abs(y0 - y0_1) <= abs(y0 - y0_2):
                    ys = 0.5 * (-1.0 * b + delta) / a
                else:
                    ys = 0.5 * (-1.0 * b - delta) / a

            pts = np.hstack((xs[:, None], ys[:, None]))

            # 将插值点旋转回去
            pts = np.matmul(pts, R)
            pts += means

            pts_list.append(pts)

        if num is not None:
            return np.asarray(pts_list)
        else:
            return pts_list

    def insert_line(self, image, lines, color, thickness=1):
        pts_list = self.interp_line(lines)
        for pts in pts_list:
            pts = np.round(pts).astype(np.int32)
            cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)

        return image
