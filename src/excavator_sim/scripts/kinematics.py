#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np


class ExcavatorKinematics:
    """
    关节顺序: [swing, boom, arm, bucket]  (单位: rad)

    约定（重要）:
    - q_boom = 0, q_arm = 0, q_bucket = 0 时:
        boom / arm / bucket 都沿 +X 方向水平伸直
    - 关节角为正: 在 x-z 平面内绕 +y 轴转动，末端向 +z 方向抬起

    接口:
    - forward_kinematics(q) -> (x, y, z)
    - inverse_kinematics([x, y, z], target_pitch=None) -> q 或 None

      target_pitch:
          铲斗尖端在 x-z 平面内相对“水平”的绝对俯仰角（世界坐标）。
          例：
              0         : 铲斗尖端朝前（水平）
              -pi/2     : 铲斗尖端竖直向下
              +pi/2     : 铲斗尖端竖直向上
          若为 None，则只做位置 IK，不约束姿态（姿态由几何顺带决定）。
    """

    def __init__(self):
        # ============= 1. 几何参数（从你原始代码抄过来） =============

        # 1.1 底盘和座舱
        self.track_radius = 0.25
        self.chassis_height = 0.4
        self.cabin_length = 1.6
        self.cabin_height = 1.0

        # chassis 几何中心相对 base_footprint 的 z 偏移:
        z_chassis_center = self.track_radius + self.chassis_height / 2.0  # 0.25 + 0.2 = 0.45

        # swing_joint origin: z = chassis_height/2 (在 chassis 坐标),
        # 所以 swing 轴心在世界里的高度:
        self.H_SWING = z_chassis_center + self.chassis_height / 2.0  # 0.45 + 0.2 = 0.65

        # boom_joint origin xyz="cabin_length/2 0 cabin_height"
        self.OFFSET_BOOM_X = self.cabin_length / 2.0  # 0.8
        self.OFFSET_BOOM_Z = self.cabin_height        # 1.0

        # boom 根部在世界坐标中的基准位置 (swing=0 时)
        self.BOOM_ROOT_BASE_X = self.OFFSET_BOOM_X             # 0.8
        self.BOOM_ROOT_BASE_Z = self.H_SWING + self.OFFSET_BOOM_Z  # 0.65 + 1.0 = 1.65

        # 1.2 连杆长度
        self.L_BOOM = 2.5
        self.L_ARM = 2.0

        # 1.3 bucket_tip 相对 bucket joint 的偏移 (URDF: 0.8, 0, -0.5)
        self.BUCKET_X_OFFSET = 0.8
        self.BUCKET_Z_OFFSET = -0.5

        # 极坐标: 尖端相对 bucket 关节距离 & 偏角
        self.L_BUCKET_EFF = math.hypot(self.BUCKET_X_OFFSET, self.BUCKET_Z_OFFSET)
        # Δφ: 从 bucket 自身的指向方向到 tip 的连线的偏角
        # 假设 bucket 自身指向“沿 +X”时，tip 落在 (BUCKET_X_OFFSET, BUCKET_Z_OFFSET)
        # 则 tip 连线的极角为:
        self.PHI_BUCKET_OFFSET = math.atan2(self.BUCKET_Z_OFFSET, self.BUCKET_X_OFFSET)

        # ============= 2. 关节极限（按你之前的习惯写一个） =============
        self.SWING_MIN = -math.pi
        self.SWING_MAX = math.pi

        # 这些值你可以根据 URDF 实际再改，这里先给一个较宽泛范围
        self.BOOM_MIN = -math.radians(15.0)   # 略微向下
        self.BOOM_MAX = math.radians(75.0)    # 抬得比较高

        self.ARM_MIN = -math.radians(120.0)   # 向后折
        self.ARM_MAX = math.radians(10.0)     # 略微抬起

        self.BUCKET_MIN = -math.radians(140.0)
        self.BUCKET_MAX = math.radians(60.0)

        # ============= 3. 若 URDF 中零位不是“水平指向前”，可在这里加偏置 =========
        # 当前假定: q_boom/q_arm/q_bucket = 0 时，三杆都水平向前，
        # 如与你实际不符，可以设置:
        self.BOOM_ZERO_OFFSET = 0.0
        self.ARM_ZERO_OFFSET = 0.0
        self.BUCKET_ZERO_OFFSET = 0.0

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(x, min_v, max_v):
        return max(min_v, min(max_v, x))

    # ------------------------------------------------------------------
    # 正运动学：给关节角 -> 算铲斗尖端在世界坐标系的位置
    # ------------------------------------------------------------------
    def forward_kinematics(self, q):
        """
        q: [q_swing, q_boom, q_arm, q_bucket]
        返回: (x, y, z) - bucket_tip 在世界坐标系中的位置
        """
        q_swing, q_boom, q_arm, q_bucket = q

        # 1) 在 swing=0 的平面 (x-z) 上做 planar FK，再绕 z 旋转 swing
        # boom 根部在世界坐标 (swing=0) 下:
        x0 = self.BOOM_ROOT_BASE_X
        z0 = self.BOOM_ROOT_BASE_Z

        # 加上零位偏置
        th_boom = q_boom + self.BOOM_ZERO_OFFSET
        th_arm  = q_arm  + self.ARM_ZERO_OFFSET
        th_bucket = q_bucket + self.BUCKET_ZERO_OFFSET

        # boom 末端
        x1 = x0 + self.L_BOOM * math.cos(th_boom)
        z1 = z0 + self.L_BOOM * math.sin(th_boom)

        # arm 末端
        th_ba = th_boom + th_arm
        x2 = x1 + self.L_ARM * math.cos(th_ba)
        z2 = z1 + self.L_ARM * math.sin(th_ba)

        # bucket 自身方向（不包含 tip 偏移）
        th_total = th_ba + th_bucket

        # tip 位置 = bucket_joint + R * [cos(th_total + PHI), sin(th_total + PHI)]
        tip_angle = th_total + self.PHI_BUCKET_OFFSET
        x_tip = x2 + self.L_BUCKET_EFF * math.cos(tip_angle)
        z_tip = z2 + self.L_BUCKET_EFF * math.sin(tip_angle)

        # 2) 再绕 z 旋转 swing 得到 (x, y)
        # 先在 local x-y 平面里: (x_tip, 0) -> 旋转 q_swing
        cos_yaw = math.cos(q_swing)
        sin_yaw = math.sin(q_swing)

        world_x = x_tip * cos_yaw
        world_y = x_tip * sin_yaw
        world_z = z_tip

        return np.array([world_x, world_y, world_z])

    # ------------------------------------------------------------------
    # 逆运动学：给铲斗尖端的世界坐标 -> 求关节角
    # ------------------------------------------------------------------
    def inverse_kinematics(self, target_pos, target_pitch=None):
        """
        target_pos: [x, y, z] 末端在世界坐标的位置
        target_pitch: 末端在 x-z 平面的绝对俯仰角 (rad)，相对水平。
                      None 表示不强制姿态，只管位置。

        返回: [q_swing, q_boom, q_arm, q_bucket] 或 None （不可达）
        """
        tx, ty, tz = target_pos

        # ---------- 1) 先解 swing ----------
        q_swing = math.atan2(ty, tx)  # 让末端投影在 x-y 平面指向目标
        # 关节限幅（连续关节这里一般无所谓）
        q_swing = self._clamp(q_swing, self.SWING_MIN, self.SWING_MAX)

        # 在 swing=0 的平面里看: 目标点在 turret 前方的投影
        r = math.hypot(tx, ty)  # 到 z 轴的水平距离 = x-y 平面半径
        x_planar = r
        z_planar = tz

        # ---------- 2) 把 “tip 目标” 转成 “bucket_joint 目标” ----------
        # 我们希望最终 tip 在 (x_planar, z_planar)
        # 若给了 target_pitch，则 bucket 的绝对方向应该是 target_pitch
        # 此时 bucket_joint 的位置 = tip - R * [cos(theta + PHI), sin(theta + PHI)]
        if target_pitch is not None:
            th_total = target_pitch  # bucket 自身在 x-z 平面内的绝对俯仰

            tip_angle = th_total + self.PHI_BUCKET_OFFSET
            x_joint = x_planar - self.L_BUCKET_EFF * math.cos(tip_angle)
            z_joint = z_planar - self.L_BUCKET_EFF * math.sin(tip_angle)
        else:
            # 不限制姿态:
            # 先粗暴假设 bucket 指向 “指向 tip 的方向”，
            # 即 th_total = atan2(z, x)，再反推 bucket_joint。
            # 这样只是为了得到一个合理的 x_joint/z_joint 位置，然后由两杆 IK 决定姿态。
            th_guess = math.atan2(z_planar - self.BOOM_ROOT_BASE_Z,
                                  x_planar - self.BOOM_ROOT_BASE_X)
            tip_angle = th_guess + self.PHI_BUCKET_OFFSET
            x_joint = x_planar - self.L_BUCKET_EFF * math.cos(tip_angle)
            z_joint = z_planar - self.L_BUCKET_EFF * math.sin(tip_angle)

        # 把 bucket_joint 的目标位置转成相对 boom_root 的坐标
        dx = x_joint - self.BOOM_ROOT_BASE_X
        dz = z_joint - self.BOOM_ROOT_BASE_Z

        # ---------- 3) 两杆 IK (boom + arm) ----------
        L1 = self.L_BOOM
        L2 = self.L_ARM

        d = math.hypot(dx, dz)
        # 超出机械臂最大/最小可达范围:
        if d > (L1 + L2) or d < abs(L1 - L2):
            return None

        # 目标点在 boom_root 坐标里的极角
        alpha = math.atan2(dz, dx)

        # 余弦定理:
        # cos(gamma) = (L1^2 + L2^2 - d^2) / (2 L1 L2)  (夹角在两杆之间)
        cos_gamma = (L1**2 + L2**2 - d**2) / (2.0 * L1 * L2)
        cos_gamma = self._clamp(cos_gamma, -1.0, 1.0)
        gamma = math.acos(cos_gamma)

        # cos(beta) = (L1^2 + d^2 - L2^2) / (2 L1 d)
        cos_beta = (L1**2 + d**2 - L2**2) / (2.0 * L1 * d)
        cos_beta = self._clamp(cos_beta, -1.0, 1.0)
        beta = math.acos(cos_beta)

        # 这里选“肘下”解：
        # boom 从水平抬起，arm 向下折一点
        th_boom = alpha - beta                 # boom 的绝对角
        th_arm_abs = math.pi - gamma           # arm 相对于 boom 的角度（标准两杆公式）

        # 把绝对角转换回 “关节角 = 实际角 - 零位偏置”
        q_boom = th_boom - self.BOOM_ZERO_OFFSET
        q_arm = th_arm_abs - self.ARM_ZERO_OFFSET

        # 超出关节极限则认为不可达
        if not (self.BOOM_MIN <= q_boom <= self.BOOM_MAX):
            return None
        if not (self.ARM_MIN <= q_arm <= self.ARM_MAX):
            return None

        # ---------- 4) 求 bucket 角 ----------
        if target_pitch is not None:
            # 总绝对俯仰：th_total = boom + arm + bucket （再加各自偏置）
            th_total_des = target_pitch

            # 实际 boom/arm 的绝对角:
            th_boom_abs = q_boom + self.BOOM_ZERO_OFFSET
            th_arm_rel = q_arm + self.ARM_ZERO_OFFSET

            # bucket 绝对角 = th_total_des - (boom + arm)
            th_bucket_abs = th_total_des - (th_boom_abs + th_arm_rel)

            # 转回关节角
            q_bucket = th_bucket_abs - self.BUCKET_ZERO_OFFSET
        else:
            # 不限制姿态: 让 bucket 角保持一个“自然姿势”
            # 比如让铲斗总是与 arm 的方向对齐 -> q_bucket = 0
            q_bucket = 0.0

        if not (self.BUCKET_MIN <= q_bucket <= self.BUCKET_MAX):
            # 有些姿态要求会挡住铲斗关节限位，这里可以回退为不限制姿态
            # 也可以直接返回 None，看你想要哪个行为
            # 这里选择回退为不限制姿态：
            q_bucket = self._clamp(q_bucket, self.BUCKET_MIN, self.BUCKET_MAX)

        return np.array([q_swing, q_boom, q_arm, q_bucket], dtype=float)


# ----------------------------------------------------------------------
# 简单自测（你可以单独跑这个文件来验证 IK/FK 是否互相对得上）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    kin = ExcavatorKinematics()

    # 随便找一组关节角
    q_test = [0.3, 0.2, -0.5, 0.1]
    p = kin.forward_kinematics(q_test)
    print("FK position:", p)

    # 用 IK 再反解回来（不限制姿态）
    q_ik = kin.inverse_kinematics(p, target_pitch=None)
    print("IK(q) ->", q_ik)

    if q_ik is not None:
        p2 = kin.forward_kinematics(q_ik)
        print("FK(IK(p)):", p2)
        print("position error:", np.linalg.norm(p2 - p))