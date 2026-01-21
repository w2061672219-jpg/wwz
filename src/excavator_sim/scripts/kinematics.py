#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np


class ExcavatorKinematics:
    """
    关节顺序: [swing, boom, arm, bucket]  (单位: rad)

    - forward_kinematics(q) -> [x, y, z]
    - inverse_kinematics([x, y, z], target_pitch=None) -> q or None
      target_pitch: 铲斗尖端在 x-z 平面内相对水平的俯仰角 (rad)，
                    若为 None，则只做位置 IK，不约束姿态。
    """

    def __init__(self):
        # ============= 1. 从 URDF 抽出来的几何参数 =============

        # 1.1 底盘和座舱
        self.track_radius = 0.25
        self.chassis_height = 0.4
        self.cabin_length = 1.6
        self.cabin_height = 1.0

        # chassis 几何中心相对 base_footprint 的 z 偏移:
        # base_joint origin: z = track_radius + chassis_height/2
        z_chassis_center = self.track_radius + self.chassis_height / 2.0  # 0.25 + 0.2 = 0.45

        # swing_joint origin: z = chassis_height/2 (在 chassis 坐标),
        # 所以 swing 轴心在世界里高度:
        self.H_SWING = z_chassis_center + self.chassis_height / 2.0  # 0.45 + 0.2 = 0.65

        # boom_joint origin xyz="cabin_length/2 0 cabin_height"
        self.OFFSET_BOOM_X = self.cabin_length / 2.0  # 0.8
        self.OFFSET_BOOM_Z = self.cabin_height        # 1.0

        # boom 根部在世界坐标中的基准位置 (swing=0 时)
        self.BOOM_ROOT_BASE_X = self.OFFSET_BOOM_X  # 0.8
        self.BOOM_ROOT_BASE_Z = self.H_SWING + self.OFFSET_BOOM_Z  # 0.65 + 1.0 = 1.65

        # 1.2 连杆长度
        self.L_BOOM = 2.5
        self.L_ARM = 2.0

        # 1.3 bucket_tip 相对 bucket joint 的偏移 (URDF: 0.8, 0, -0.5)
        self.BUCKET_X_OFFSET = 0.8
        self.BUCKET_Z_OFFSET = -0.5

        # 极坐标: 尖端相对 bucket 关节距离 & 偏角
        self.L_BUCKET_EFF = math.hypot(self.BUCKET_X_OFFSET, self.BUCKET_Z_OFFSET)
        # 在 arm/bucket 所在的 x-z 平面中的夹角: atan2(z, x)
        self.PHI_BUCKET_OFFSET = math.atan2(self.BUCKET_Z_OFFSET, self.BUCKET_X_OFFSET)

        # ============= 2. 关节极限 (从 URDF limit) =============

        # swing_joint continuous，这里给一个[-pi, pi] 用作 clamp 即可
        self.LIMITS = {
            "swing": (-math.pi, math.pi),       # continuous
            "boom": (-1.5, 0.5),
            "arm": (-1.0, 2.5),
            "bucket": (-1.5, 1.5),
        }

    # ---------------------------------------------------------
    # 工具函数
    # ---------------------------------------------------------
    def clamp_angle(self, angle, joint_name):
        mn, mx = self.LIMITS[joint_name]
        if angle < mn:
            return mn
        if angle > mx:
            return mx
        return angle

    # ---------------------------------------------------------
    # 正运动学: q = [swing, boom, arm, bucket] -> [x, y, z]
    # ---------------------------------------------------------
    def forward_kinematics(self, joints):
        """
        :param joints: [q_swing, q_boom, q_arm, q_bucket]  (rad)
        :return: [x, y, z] bucket_tip 在 world 坐标系中的位置 (m)
        """
        q_swing, q_boom, q_arm, q_bucket = joints

        # 1) boom 根在世界坐标下的位置:
        #    先考虑 swing = 0 时，boom 根在 world 是 (BOOM_ROOT_BASE_X, 0, BOOM_ROOT_BASE_Z)
        #    swing 转动是绕 z 轴转，把 (x,0) 旋转到 (x*cos, x*sin)
        cos_sw = math.cos(q_swing)
        sin_sw = math.sin(q_swing)
        boom_root_x = cos_sw * self.BOOM_ROOT_BASE_X
        boom_root_y = sin_sw * self.BOOM_ROOT_BASE_X
        boom_root_z = self.BOOM_ROOT_BASE_Z

        # 2) 在"前方平面"中建立局部坐标 (r, z)，r 是从 swing 轴沿前方方向的水平距离。
        #    注意: 在 swing 后的 world 坐标中，前方方向是 swing 旋转后的 x 方向，
        #    但在这个平面分析里，我们只关心长度，不管 y 。
        #    由于 boom/root 在 turret 前方，所以 r_root 就是 BOOM_ROOT_BASE_X
        r_root = self.BOOM_ROOT_BASE_X
        z_root = self.BOOM_ROOT_BASE_Z

        # 3) boom 末端 (arm_joint) 在 平面 (r,z) 里的坐标:
        #    约定: boom 角 q_boom = 0 时是水平向前；正方向是"抬起" (绕 y 轴转)。
        r_boom_end = r_root + self.L_BOOM * math.cos(q_boom)
        z_boom_end = z_root + self.L_BOOM * math.sin(q_boom)

        # 4) arm 末端 (bucket_joint) 的坐标:
        q_ba = q_boom + q_arm
        r_arm_end = r_boom_end + self.L_ARM * math.cos(q_ba)
        z_arm_end = z_boom_end + self.L_ARM * math.sin(q_ba)

        # 5) bucket_tip 的坐标:
        #    bucket 尖端相对 bucket_joint 有一个固定偏置:
        #    在当下关节配置下，铲斗整体方向角度 = q_boom + q_arm + q_bucket
        q_total = q_ba + q_bucket  # boom+arm+bucket
        theta_tip = q_total + self.PHI_BUCKET_OFFSET

        r_tip = r_arm_end + self.L_BUCKET_EFF * math.cos(theta_tip)
        z_tip = z_arm_end + self.L_BUCKET_EFF * math.sin(theta_tip)

        # 6) 把 (r_tip, z_tip) + swing 角，转成 (x,y,z) 世界坐标。
        #    由于 r 是绕 swing 轴的半径，在 world 中:
        #    x = r * cos(q_swing), y = r * sin(q_swing)
        x = r_tip * cos_sw
        y = r_tip * sin_sw
        z = z_tip

        return [x, y, z]

    # ---------------------------------------------------------
    # 逆运动学: [x, y, z] (+ 期望俯仰) -> q = [swing, boom, arm, bucket]
    # ---------------------------------------------------------
    def inverse_kinematics(self, target_pos, target_pitch=None):
        """
        逆运动学:
        :param target_pos: [x, y, z] bucket_tip 世界坐标 (m)
        :param target_pitch: 希望铲斗尖端在 x-z 平面中的俯仰角 (相对水平，rad)。
                             若为 None，则只满足位置，不强制姿态。
        :return: [q_swing, q_boom, q_arm, q_bucket] or None (不可达)
        """
        x, y, z = target_pos

        # 1) 先求 swing 角: 在 x-y 平面上的方位角
        #    swing 绕 z 轴，是从 x 轴朝 y 轴旋转，右手规则。
        q_swing = math.atan2(y, x)  # [-pi, pi]

        # 2) 把目标投影到 swing 前方平面的 (r,z)
        #    半径 r = sqrt(x^2 + y^2)
        r_target_world = math.hypot(x, y)

        # boom 根在该平面内的坐标:
        r_root = self.BOOM_ROOT_BASE_X
        z_root = self.BOOM_ROOT_BASE_Z

        # 目标相对 boom 根的坐标 (这是 tip 的位置，不是 bucket_joint)
        r_tip_rel = r_target_world - r_root
        z_tip_rel = z - z_root

        # 3) 两杆 IK 用的是"bucket_tip"的位置，但两杆实际上连到的是 bucket_joint。
        #    所以要先把 tip 的偏置去掉，算出 bucket_joint 的目标位置。
        #    这里牵涉到姿态，如果不指定 target_pitch，只能先“猜”一个方向再调。
        if target_pitch is None:
            # 简单策略：先假设 total 角度大致指向目标 (r_tip_rel,z_tip_rel) 的方向
            theta_guess = math.atan2(z_tip_rel, r_tip_rel)
            # 把 tip 偏置往反方向平移回 bucket_joint
            r_joint = r_tip_rel - self.L_BUCKET_EFF * math.cos(theta_guess + self.PHI_BUCKET_OFFSET)
            z_joint = z_tip_rel - self.L_BUCKET_EFF * math.sin(theta_guess + self.PHI_BUCKET_OFFSET)
            # 后面用 (r_joint,z_joint) 做两杆 IK
            enforce_pitch = False
        else:
            # 有明确俯仰角需求:
            # total 角度（铲斗尖端连杆方向） = target_pitch
            # tip 偏置相对 total 的角度已经在 PHI_BUCKET_OFFSET 中
            total_angle = target_pitch
            r_joint = r_tip_rel - self.L_BUCKET_EFF * math.cos(total_angle + self.PHI_BUCKET_OFFSET)
            z_joint = z_tip_rel - self.L_BUCKET_EFF * math.sin(total_angle + self.PHI_BUCKET_OFFSET)
            enforce_pitch = True

        # 4) 两杆 IK: 从 boom_root 指到 bucket_joint 的矢量
        #    长度 d = sqrt(r_joint^2 + z_joint^2)
        d = math.hypot(r_joint, z_joint)
        L1 = self.L_BOOM
        L2 = self.L_ARM

        # 超出可达工作空间
        if d > (L1 + L2) or d < abs(L1 - L2):
            return None

        # 利用余弦定理求 arm 角 (在平面里，q_arm 相对 boom)
        # cos(gamma) = (L1^2 + L2^2 - d^2) / (2 L1 L2)
        cos_gamma = (L1**2 + L2**2 - d**2) / (2.0 * L1 * L2)
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        gamma = math.acos(cos_gamma)  # 肘上/肘下会对应 gamma 和 -gamma 两解，这里先取一支

        # 通常机器人关节的标准形式: q_arm = pi - gamma (视你定义而定)
        # 这里 boom和arm是串联，两杆 IK 常用: q_arm = pi - gamma 或 -(pi - gamma)
        # 我们看几何: 当 boom 水平(q_boom=0), arm 向前伸直时，希望 q_arm=0。
        # 此时 d = L1 + L2, cos_gamma = 1, gamma=0 => 如果 q_arm = pi - gamma = pi (不对)
        # 因此改成: q_arm = gamma - pi，则:
        #   d = L1 + L2 => gamma=0 => q_arm = -pi (不对)
        # 所以更简单: 直接把 q_arm 当作"内角差"，用下面方法统一求:

        # 先求目标连线方向角
        alpha = math.atan2(z_joint, r_joint)

        # cos(beta) = (L1^2 + d^2 - L2^2)/(2 L1 d)
        cos_beta = (L1**2 + d**2 - L2**2) / (2.0 * L1 * d)
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)

        # 两种构型: 肘上 / 肘下，对应 boom 角不同:
        # 这里先取“肘下”(挖掘机常见的形态)，boom角:
        q_boom = alpha - beta
        # total angle at bucket_joint:
        q_ba = alpha + beta  # 这是连线方向相对boom root的另一种写法，也可以 L1,L2推导

        # 再通过 L1,L2 关系求 q_arm = q_ba - q_boom
        q_arm = q_ba - q_boom

        # 5) 有无末端姿态约束决定 bucket 角
        if enforce_pitch:
            # total angle q_total = q_boom + q_arm + q_bucket 应该约等于 target_pitch
            q_total = target_pitch
            q_bucket = q_total - (q_boom + q_arm)
        else:
            # 若不强制末端俯仰，则让 bucket 保持某个“默认”角度，
            # 例如让 bucket 尖端连线方向 roughly 和 bucket_joint->tip 方向一致，只追位置。
            # 这里简单取: 使 total 角等于 (r_joint,z_joint) 指向角 alpha
            q_total = alpha  # 也可以根据你希望的姿态策略改
            q_bucket = q_total - (q_boom + q_arm)

        # 6) clamp 到 URDF 关节范围
        q_swing = self.clamp_angle(q_swing, "swing")
        q_boom = self.clamp_angle(q_boom, "boom")
        q_arm = self.clamp_angle(q_arm, "arm")
        q_bucket = self.clamp_angle(q_bucket, "bucket")

        return [q_swing, q_boom, q_arm, q_bucket]


# ---------------------------------------------------------
# 自测: 随机点 FK -> IK -> FK 闭环
# ---------------------------------------------------------
if __name__ == "__main__":
    kin = ExcavatorKinematics()

    np.set_printoptions(precision=3, suppress=True)

    for i in range(20):
        # 在一个大致合理的区域随便采样目标点:
        # x 在 [1.0, 5.0] 之间, y 在 [-2, 2], z 在 [0.0, 3.0]
        x = float(np.random.uniform(1.0, 5.0))
        y = float(np.random.uniform(-2.0, 2.0))
        z = float(np.random.uniform(0.0, 3.0))
        target = [x, y, z]

        # 给一个大致向下的俯仰角，看看能不能解:
        target_pitch = -0.7  # 铲斗稍微向下

        q = kin.inverse_kinematics(target, target_pitch=target_pitch)
        if q is None:
            print("目标不可达:", np.round(target, 3))
            continue

        pos = kin.forward_kinematics(q)
        err = math.dist(target, pos)

        print(f"target={np.round(target,3)}, "
              f"q={np.round(q,3)}, "
              f"fk={np.round(pos,3)}, "
              f"err={err:.4f}")