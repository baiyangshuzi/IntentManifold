# -*- coding: utf-8 -*-
"""v0.69 多层级状态估计器（Kalman）+ 误差反馈控制器（PID）——推理期闭环控制

架构（用户方案）：
  观测：句元级 sent_proj 序列 z(t)
  Kalman2D：状态 x=[intent, drift_speed]——平滑估计 x_est + 先验预测 x_pred（前馈）
  PID：e_para=Kp 比例（段级误差）/ ∫e_doc 积分（篇幅级累积）/ de 微分（句元变化率）
  执行：β = clamp(Kp·e_para + Ki·∫e_doc + Kd·de + α_feed·e_pred, 0, β_max)
纯函数——可单测（模拟序列收敛验证）
"""
import numpy as np


class Kalman2D:
    """2 维卡尔曼：状态 [intent, drift_speed]——观测 intent（含噪声）"""

    def __init__(self, q=0.001, r=0.0025, x0=0.85):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])   # intent += drift
        self.H = np.array([[1.0, 0.0]])                # 观测 intent
        self.Q = np.eye(2) * q
        self.R = np.array([[r]])
        self.x = np.array([[x0], [0.0]])               # 初始意图估计（AI 基线）
        self.P = np.eye(2) * 0.01
        self.x_pred = self.x.copy()

    def predict(self):
        """v0.73-2 段开始调用：把上一段后验前推一步——返回预测本段状态的先验
        （修复 v0.69-3 实现问题：段开始读旧 x_pred 是"预测上一段"的先验——缺一步 predict(后验)）"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x_pred = self.x.copy()
        return float(self.x_pred[0, 0])

    def step(self, z):
        """观测 z（句元投影）→ 返回 (x_est, x_pred)——predict+update 合并（兼容旧调用）"""
        # predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x_pred = self.x.copy()
        # update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S[0, 0]
        innov = z - (self.H @ self.x)[0, 0]
        self.x = self.x + K * innov
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x_pred[0, 0])


class PIDLoop:
    """四层级 PID：Kp 比例（段级）/Ki 积分（篇幅级累积）/Kd 微分（句元变化率）+ 前馈

    β(t) = clamp(Kp·e_para + Ki·∫e_doc + Kd·de + α_feed·e_pred, 0, β_max)
    """

    def __init__(self, kp=12.0, ki=0.3, kd=6.0, alpha_feed=0.3,
                 target=0.90, beta_max=2.0, int_limit=0.3):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.alpha_feed = alpha_feed
        self.target = target
        self.beta_max = beta_max
        self.int_limit = int_limit
        self.int_acc = 0.0          # 篇幅级积分累积
        self.prev_x = None          # 句元变化率基线
        self.feed_triggers = 0      # 前馈触发次数（K4 判据）
        self.trace = []             # 控制轨迹 [(t, z, x_est, e_para, de, beta)]

    def reset(self):
        self.int_acc = 0.0
        self.prev_x = None
        self.feed_triggers = 0
        self.trace = []

    def update(self, z, x_est, x_pred, t):
        """每句元调用——返回动态 β（并记录轨迹）"""
        e_para = self.target - x_est
        # 积分（篇幅级——跨段累积——限幅防 windup）
        self.int_acc += e_para
        self.int_acc = np.clip(self.int_acc, -self.int_limit, self.int_limit)
        # 微分（句元变化率）
        de = 0.0 if self.prev_x is None else (x_est - self.prev_x)
        self.prev_x = x_est
        # 前馈（Kalman 预测——预判漂移增大→提前加 β——K4）
        e_pred = self.target - x_pred
        feed = 0.0
        if e_pred > e_para + 0.02:      # 预测下一步误差增大（漂移加速）
            feed = self.alpha_feed * e_pred
            self.feed_triggers += 1
        beta = self.kp * e_para + self.ki * self.int_acc + self.kd * de + feed
        beta = float(np.clip(beta, 0.0, self.beta_max))
        self.trace.append((t, float(z), float(x_est), float(e_para), float(de), beta))
        return beta


def selftest():
    """单测：合成序列（真值 0.82→漂移 0.78——含噪声）——Kalman 估计收敛 + PID β 自适应"""
    rng = np.random.default_rng(1)
    truth = np.linspace(0.82, 0.78, 60)
    obs = truth + rng.normal(0, 0.05, 60)
    kf = Kalman2D(q=0.001, r=0.0025)
    pid = PIDLoop()
    errs = []
    betas = []
    for t, z in enumerate(obs):
        x_est, x_pred = kf.step(z)
        beta = pid.update(z, x_est, x_pred, t)
        errs.append(abs(x_est - truth[t]))
        betas.append(beta)
    print(f'单测：估计 MSE {np.mean(errs):.4f} vs 观测噪声 0.05'
          f'（平滑有效（估计优于观测）{"✓" if np.mean(errs) < 0.05 else "✗"}——漂移序列平滑滞后为正常）')
    print(f'β 自适应：低误差段均值 {np.mean(betas[:20]):.3f} vs 漂移后 {np.mean(betas[30:]):.3f}'
          f'（漂移后升高 {"✓" if np.mean(betas[30:]) > np.mean(betas[:20]) else "✗"}）')
    print(f'前馈触发 {pid.feed_triggers} 次（K4 判据）')
    return np.mean(errs) < 0.05


if __name__ == '__main__':
    selftest()
