#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np


class ExcavatorKinematics:
    """
    关节顺序: [swing, boom, arm, bucket]  (单位: rad)
    
    坐标系约定：
    - swing = 0 时，机械臂沿 +X 方向伸展
    - swing 正方向：从上往下看逆时针旋转（Y正方向）
    """

    def __init__(self):
        # ============= 1. 几何参数 =============
        self.track_radius = 0.25
        self.chassis_height = 0.4
        self.cabin_length = 1.6
        self.cabin_height = 1.0

        z_chassis_center = self.track_radius + self.chassis_height / 2.0
        self.H_SWING = z_chassis_center + self.chassis_height / 2.0

        self.OFFSET_BOOM_X = self.cabin_length / 2.0
        self.OFFSET_BOOM_Z = self.cabin_height

        self.BOOM_ROOT_BASE_X = self.OFFSET_BOOM_X
        self.BOOM_ROOT_BASE_Z = self.H_SWING + self.OFFSET_BOOM_Z

        # 连杆长度
        self.L_BOOM = 2.5
        self.L_ARM = 2.0

        # bucket_tip 相对 bucket joint 的偏移 (URDF: 0.8, 0, -0.5)
        self.BUCKET_X_OFFSET = 0.8
        self.BUCKET_Z_OFFSET = -0.5

        # 等效长度和角度偏移
        self.L_BUCKET_EFF = math.hypot(self.BUCKET_X_OFFSET, self.BUCKET_Z_OFFSET)
        self.PHI_BUCKET_OFFSET = math.atan2(self.BUCKET_Z_OFFSET, self.BUCKET_X_OFFSET)

        # ============= 2. 关节极限 =============
        self.SWING_MIN = -math.pi
        self.SWING_MAX = math.pi

        self.BOOM_MIN = -math.radians(30.0)
        self.BOOM_MAX = math.radians(90.0)

        self.ARM_MIN = -math.radians(160.0)
        self.ARM_MAX = math.radians(60.0)

        self.BUCKET_MIN = -math.radians(180.0)
        self.BUCKET_MAX = math.radians(120.0)

    @staticmethod
    def _clamp(x, min_v, max_v):
        return max(min_v, min(max_v, x))

    def _check_joint_limits(self, q_bab):
        """检查 [boom, arm, bucket] 是否在限制范围内"""
        q_boom, q_arm, q_bucket = q_bab
        
        if not (self.BOOM_MIN - 0.01 <= q_boom <= self.BOOM_MAX + 0.01):
            return False
        if not (self.ARM_MIN - 0.01 <= q_arm <= self.ARM_MAX + 0.01):
            return False
        if not (self.BUCKET_MIN - 0.01 <= q_bucket <= self.BUCKET_MAX + 0.01):
            return False
        return True

    def forward_kinematics(self, q):
        """
        正运动学计算
        q: [q_swing, q_boom, q_arm, q_bucket]
        返回: (x, y, z) - 铲斗尖端在世界坐标系中的位置
        """
        q_swing, q_boom, q_arm, q_bucket = q

        # boom 根部在 swing 旋转后的局部坐标系中的位置
        # （局部坐标系：沿 swing 方向为 r 轴，垂直向上为 z 轴）
        r0 = self.BOOM_ROOT_BASE_X
        z0 = self.BOOM_ROOT_BASE_Z

        # boom 末端位置（在 r-z 平面内）
        r_boom_end = r0 + self.L_BOOM * math.cos(q_boom)
        z_boom_end = z0 + self.L_BOOM * math.sin(q_boom)

        # arm 累积角度（相对水平面）
        theta_arm = q_boom + q_arm
        r_arm_end = r_boom_end + self.L_ARM * math.cos(theta_arm)
        z_arm_end = z_boom_end + self.L_ARM * math.sin(theta_arm)

        # bucket 累积角度（相对水平面）
        theta_bucket = theta_arm + q_bucket
        
        # bucket tip 位置
        r_tip = r_arm_end + self.L_BUCKET_EFF * math.cos(theta_bucket + self.PHI_BUCKET_OFFSET)
        z_tip = z_arm_end + self.L_BUCKET_EFF * math.sin(theta_bucket + self.PHI_BUCKET_OFFSET)

        # 转换到世界坐标（考虑 swing 旋转）
        x = r_tip * math.cos(q_swing)
        y = r_tip * math.sin(q_swing)
        z = z_tip

        return x, y, z

    def inverse_kinematics(self, target_pos, target_pitch=None):
        """
        逆运动学求解
        
        关键修正：正确处理 swing 角度和径向距离的关系
        """
        x, y, z = target_pos

        # ========== 1. 求解 swing 角 ==========
        # swing 角度 = 目标点在 XY 平面上的方位角
        r_target_xy = math.hypot(x, y)
        
        if r_target_xy < 0.01:
            # 目标点在 Z 轴上，swing 可以任意，设为 0
            q_swing = 0.0
            r_target = abs(x)  # 实际上这种情况需要特殊处理
        else:
            q_swing = math.atan2(y, x)
            # 目标点在 swing 旋转后的 r-z 平面内的径向坐标
            # 就是目标点到 Z 轴的水平距离
            r_target = r_target_xy

        # ========== 2. 在 r-z 平面内求解 ==========
        # 现在问题简化为：在 r-z 平面内，让 tip 到达 (r_target, z)
        
        if target_pitch is None:
            target_pitch = -0.5

        # bucket 绝对姿态角（相对于水平面）
        theta_bucket_abs = target_pitch
        
        # 从 tip 反推 arm joint 位置（在 r-z 平面内）
        angle_to_tip = theta_bucket_abs + self.PHI_BUCKET_OFFSET
        r_arm_joint = r_target - self.L_BUCKET_EFF * math.cos(angle_to_tip)
        z_arm_joint = z - self.L_BUCKET_EFF * math.sin(angle_to_tip)

        # ========== 3. 两杆 IK: boom + arm ==========
        # boom 根部位置（在 r-z 平面内）
        r0 = self.BOOM_ROOT_BASE_X
        z0 = self.BOOM_ROOT_BASE_Z
        
        # arm joint 相对于 boom 根部的位置
        r_rel = r_arm_joint - r0
        z_rel = z_arm_joint - z0

        L1 = self.L_BOOM
        L2 = self.L_ARM
        d = math.hypot(r_rel, z_rel)

        # 检查可达性
        if d > L1 + L2 - 0.001:
            return self._try_alternative_pitch(target_pos, target_pitch)
        if d < abs(L1 - L2) + 0.001:
            return self._try_alternative_pitch(target_pos, target_pitch)

        # 余弦定理求两杆夹角
        cos_gamma = (L1**2 + L2**2 - d**2) / (2.0 * L1 * L2)
        cos_gamma = self._clamp(cos_gamma, -1.0, 1.0)
        gamma = math.acos(cos_gamma)

        # boom 角度计算
        alpha = math.atan2(z_rel, r_rel)
        cos_beta = (L1**2 + d**2 - L2**2) / (2.0 * L1 * d)
        cos_beta = self._clamp(cos_beta, -1.0, 1.0)
        beta = math.acos(cos_beta)

        # 尝试两种解（肘上/肘下）
        solutions = []
        
        # 解1: 肘下（通常是挖掘机的正常工作姿态）
        q_boom_1 = alpha - beta
        q_arm_1 = math.pi - gamma
        theta_arm_1 = q_boom_1 + q_arm_1
        q_bucket_1 = theta_bucket_abs - theta_arm_1
        
        if self._check_joint_limits([q_boom_1, q_arm_1, q_bucket_1]):
            solutions.append([q_swing, q_boom_1, q_arm_1, q_bucket_1])

        # 解2: 肘上
        q_boom_2 = alpha + beta
        q_arm_2 = -(math.pi - gamma)
        theta_arm_2 = q_boom_2 + q_arm_2
        q_bucket_2 = theta_bucket_abs - theta_arm_2
        
        if self._check_joint_limits([q_boom_2, q_arm_2, q_bucket_2]):
            solutions.append([q_swing, q_boom_2, q_arm_2, q_bucket_2])

        if solutions:
            # 选择 arm 角度更接近 0 的解（更自然的姿态）
            best = min(solutions, key=lambda s: abs(s[2]))
            return best

        return self._try_alternative_pitch(target_pos, target_pitch)

    def _try_alternative_pitch(self, target_pos, original_pitch):
        """尝试不同的 pitch 值来找到可行解"""
        x, y, z = target_pos
        r_target_xy = math.hypot(x, y)
        q_swing = math.atan2(y, x) if r_target_xy > 0.01 else 0.0
        r_target = r_target_xy if r_target_xy > 0.01 else abs(x)

        pitch_candidates = np.linspace(-1.5, 0.5, 21)
        
        best_solution = None
        best_error = float('inf')
        
        for pitch in pitch_candidates:
            if original_pitch is not None and abs(pitch - original_pitch) < 0.01:
                continue

            theta_bucket_abs = pitch
            angle_to_tip = theta_bucket_abs + self.PHI_BUCKET_OFFSET
            r_arm_joint = r_target - self.L_BUCKET_EFF * math.cos(angle_to_tip)
            z_arm_joint = z - self.L_BUCKET_EFF * math.sin(angle_to_tip)

            r_rel = r_arm_joint - self.BOOM_ROOT_BASE_X
            z_rel = z_arm_joint - self.BOOM_ROOT_BASE_Z

            L1 = self.L_BOOM
            L2 = self.L_ARM
            d = math.hypot(r_rel, z_rel)

            if d > L1 + L2 - 0.001 or d < abs(L1 - L2) + 0.001:
                continue

            cos_gamma = (L1**2 + L2**2 - d**2) / (2.0 * L1 * L2)
            cos_gamma = self._clamp(cos_gamma, -1.0, 1.0)
            gamma = math.acos(cos_gamma)

            alpha = math.atan2(z_rel, r_rel)
            cos_beta = (L1**2 + d**2 - L2**2) / (2.0 * L1 * d)
            cos_beta = self._clamp(cos_beta, -1.0, 1.0)
            beta = math.acos(cos_beta)

            for sign in [-1, 1]:
                q_boom = alpha + sign * beta
                q_arm = -sign * (math.pi - gamma)
                theta_arm = q_boom + q_arm
                q_bucket = theta_bucket_abs - theta_arm
                
                if self._check_joint_limits([q_boom, q_arm, q_bucket]):
                    q_full = [q_swing, q_boom, q_arm, q_bucket]
                    p_check = self.forward_kinematics(q_full)
                    error = math.sqrt((p_check[0] - x)**2 + 
                                     (p_check[1] - y)**2 + 
                                     (p_check[2] - z)**2)
                    
                    if error < best_error:
                        best_error = error
                        best_solution = q_full

        if best_solution is not None and best_error < 0.1:
            return best_solution
        
        return None


if __name__ == "__main__":
    kin = ExcavatorKinematics()
    
    print("=== 测试 swing 功能 ===")
    test_points = [
        [5.0, 0.0, 0.5],    # Y=0, swing 应该是 0
        [5.0, 1.0, 0.5],    # Y>0, swing 应该是正的
        [5.0, -1.0, 0.5],   # Y<0, swing 应该是负的
        [4.0, 2.0, 0.5],    # 较大的 Y 值
    ]
    
    for pt in test_points:
        q = kin.inverse_kinematics(pt, target_pitch=-0.6)
        if q:
            p_fk = kin.forward_kinematics(q)
            error = math.sqrt(sum((a-b)**2 for a,b in zip(p_fk, pt)))
            print(f"目标: {pt}")
            print(f"  IK: swing={math.degrees(q[0]):.1f}°, boom={math.degrees(q[1]):.1f}°, "
                  f"arm={math.degrees(q[2]):.1f}°, bucket={math.degrees(q[3]):.1f}°")
            print(f"  FK: ({p_fk[0]:.3f}, {p_fk[1]:.3f}, {p_fk[2]:.3f}), 误差={error:.4f}m")
        else:
            print(f"目标: {pt} -> IK失败!")
        print()