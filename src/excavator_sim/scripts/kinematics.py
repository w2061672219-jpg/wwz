#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import PyKDL

class ExcavatorKinematics:
    def __init__(self):
        # ==========================================
        # 1. 依据 URDF 构建 KDL Chain
        # ==========================================
        self.chain = PyKDL.Chain()

        # --- 基础偏移 ---
        # URDF: base_footprint -> chassis -> turret
        # total_z = track_radius(0.25) + chassis_height/2(0.2) + offset_to_turret(0.2) = 0.65
        self.base_height_offset = 0.65 

        # --- Segment 1: Swing (回转) ---
        # Joint: Rotate Z (Axis: 0, 0, 1)
        # Frame: 从 Swing 关节中心 -> Boom 关节中心
        # URDF: <origin xyz="0.8 0 1.0" ... /> (相对 Turret 中心)
        swing_to_boom_vec = PyKDL.Vector(0.8, 0.0, 1.0)
        self.chain.addSegment(PyKDL.Segment(
            "turret",
            PyKDL.Joint("swing_joint", PyKDL.Joint.RotZ),
            PyKDL.Frame(swing_to_boom_vec)
        ))

        # --- Segment 2: Boom (大臂) ---
        # Joint: Rotate Y (Axis: 0, 1, 0)
        # Frame: 从 Boom 关节 -> Arm 关节
        # URDF: <origin xyz="2.5 0 0" ... /> (boom_length)
        boom_to_arm_vec = PyKDL.Vector(2.5, 0.0, 0.0)
        self.chain.addSegment(PyKDL.Segment(
            "boom",
            PyKDL.Joint("boom_joint", PyKDL.Joint.RotY),
            PyKDL.Frame(boom_to_arm_vec)
        ))

        # --- Segment 3: Arm (小臂) ---
        # Joint: Rotate Y (Axis: 0, 1, 0)
        # Frame: 从 Arm 关节 -> Bucket 关节
        # URDF: <origin xyz="2.0 0 0" ... /> (arm_length)
        arm_to_bucket_vec = PyKDL.Vector(2.0, 0.0, 0.0)
        self.chain.addSegment(PyKDL.Segment(
            "arm",
            PyKDL.Joint("arm_joint", PyKDL.Joint.RotY),
            PyKDL.Frame(arm_to_bucket_vec)
        ))

        # --- Segment 4: Bucket (铲斗) ---
        # Joint: Rotate Y (Axis: 0, 1, 0)
        # Frame: 从 Bucket 关节 -> Bucket Tip (铲斗尖端)
        # URDF: <origin xyz="0.8 0 -0.5" ... />
        bucket_to_tip_vec = PyKDL.Vector(0.8, 0.0, -0.5)
        self.chain.addSegment(PyKDL.Segment(
            "bucket",
            PyKDL.Joint("bucket_joint", PyKDL.Joint.RotY),
            PyKDL.Frame(bucket_to_tip_vec)
        ))

        # ==========================================
        # 2. 依据 URDF 设置关节极限
        # ==========================================
        num_joints = self.chain.getNrOfJoints()
        self.q_min = PyKDL.JntArray(num_joints)
        self.q_max = PyKDL.JntArray(num_joints)

        # 索引对应顺序: [Swing, Boom, Arm, Bucket]
        
        # 1. Swing: URDF type="continuous", 但通常物理上设为 -pi 到 pi
        self.q_min[0] = -math.pi
        self.q_max[0] = math.pi

        # 2. Boom: <limit lower="-1.5" upper="0.5" .../>
        self.q_min[1] = -1.5
        self.q_max[1] = 0.5

        # 3. Arm: <limit lower="-1.0" upper="2.5" .../>
        self.q_min[2] = -1.0
        self.q_max[2] = 2.5

        # 4. Bucket: <limit lower="-1.5" upper="1.5" .../>
        self.q_min[3] = -1.5
        self.q_max[3] = 1.5

        # ==========================================
        # 3. 初始化求解器
        # ==========================================
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.ik_v_solver = PyKDL.ChainIkSolverVel_pinv(self.chain)
        # 使用带关节限制的牛顿-拉夫逊迭代法求解 IK
        self.ik_solver = PyKDL.ChainIkSolverPos_NR_JL(
            self.chain, 
            self.q_min, 
            self.q_max, 
            self.fk_solver, 
            self.ik_v_solver
        )

    def forward_kinematics(self, joint_angles):
        """
        正运动学: 关节角 -> 铲斗尖端坐标 (Base Footprint Frame)
        :param joint_angles: list [swing, boom, arm, bucket] (rad)
        :return: [x, y, z] (meters)
        """
        q_in = PyKDL.JntArray(4)
        for i, q in enumerate(joint_angles):
            q_in[i] = q
        
        end_frame = PyKDL.Frame()
        status = self.fk_solver.JntToCart(q_in, end_frame)
        
        if status >= 0:
            pos = end_frame.p
            # 注意：KDL Chain 是从 Swing 关节开始算的 (z=0.65m 处)
            # 如果要返回相对于地面的坐标，需要加上 base_height_offset
            return [pos.x(), pos.y(), pos.z() + self.base_height_offset]
        else:
            print("FK Error")
            return None

    def inverse_kinematics(self, target_pos, init_guess=None):
        """
        逆运动学: 目标坐标 -> 关节角
        :param target_pos: list [x, y, z] (meters, Base Footprint Frame)
        :param init_guess: list [q1, q2, q3, q4] 初始猜测值，默认为全0
        :return: list [swing, boom, arm, bucket] or None
        """
        # 1. 目标位置处理 (转为相对于 Swing 根部的坐标)
        tgt_x, tgt_y, tgt_z = target_pos
        kdl_target_vec = PyKDL.Vector(tgt_x, tgt_y, tgt_z - self.base_height_offset)
        
        # 挖掘机末端通常只关心位置(3DOF)，但我们有4个关节(Swing控制平面方向)。
        # KDL 需要一个目标 Frame (位置+姿态)。
        # 对于挖掘机，我们通常不严格约束末端的 Roll/Pitch (除非你想铲平地面)。
        # 这里我们构建一个简单的目标 Frame，位置是确定的。
        # 注意：NR_JL 求解器会尝试同时匹配位置和姿态，这对于 4-DOF 机械臂匹配 6-DOF 目标可能会失败。
        # 
        # **优化策略**: 
        # 挖掘机 IK 的核心逻辑：
        # 1. Swing (q0) 由 x,y 直接决定。
        # 2. 剩下的 Boom, Arm, Bucket 形成一个 3-link planar 链条。
        # 
        # 由于 PyKDL 的通用求解器在自由度不足(4DOF vs 6DOF)时很难收敛，
        # 建议：如果不需要指定铲斗姿态（即铲斗随动），我们只需确保位置到达。
        # 但 PyKDL IK solver 是全姿态求解。
        
        # 为了让 PyKDL 工作，我们需要一个更合适的“初始猜测”和“目标姿态”。
        # 此处为了简单演示 PyKDL 用法，我们假设一个目标姿态（例如铲斗水平）。
        # 在实际工程中，挖掘机通常使用“解析法+数值微调”或者将 Swing 分离计算。
        
        target_frame = PyKDL.Frame(kdl_target_vec)
        
        # 初始猜测
        q_init = PyKDL.JntArray(4)
        if init_guess:
            for i, val in enumerate(init_guess):
                q_init[i] = val
        else:
            # 智能猜测：Swing 角由 atan2(y, x) 决定
            q_init[0] = math.atan2(tgt_y, tgt_x)
            
        q_out = PyKDL.JntArray(4)
        
        # 调用求解器
        # 注意：这里可能因为姿态约束无法满足而返回错误，
        # 对于 4自由度机械臂，建议仅控制 Position，或使用自定义 IK。
        # 但 PyKDL 标准库是 Pos+Rot 的。
        ret = self.ik_solver.CartToJnt(q_init, target_frame, q_out)
        
        if ret >= 0:
            return [q_out[0], q_out[1], q_out[2], q_out[3]]
        else:
            # PyKDL 数值解如果在 4DOF 下强行解 6DOF 目标很容易失败
            print(f"IK Solver failed to converge (Error code: {ret})")
            return None

# ================= 测试代码 =================
if __name__ == "__main__":
    excavator = ExcavatorKinematics()
    
    # 1. 测试正运动学 (FK)
    # 设想一个姿态：Swing=0, Boom=0 (平举?), Arm=0, Bucket=0
    # 根据 URDF Limits: Boom=0 是允许的, Arm=0 允许, Bucket=0 允许
    test_joints = [0.0, 0.0, 0.0, 0.0] 
    
    fk_res = excavator.forward_kinematics(test_joints)
    print(f"关节角: {test_joints}")
    print(f"FK 结果 (x,y,z): {np.round(fk_res, 3)}")
    
    # 手算验证 (Swing=0):
    # BaseZ (0.65)
    # + Turret->BoomZ (1.0) = 1.65
    # + Turret->BoomX (0.8)
    # + Boom (2.5) + Arm (2.0) + BucketX (0.8) = 6.1
    # + BucketZ (-0.5)
    # 预期 Z = 1.65 - 0.5 = 1.15
    # 预期 X = 0.8 + 2.5 + 2.0 + 0.8 = 6.1
    # 结果应为 [6.1, 0, 1.15]
    
    # 2. 测试逆运动学 (IK)
    # 尝试解算刚才算出来的坐标
    target_pos = [6.1, 0.0, 1.15]
    print(f"\n尝试 IK 求解目标: {target_pos}")
    
    # 给予一个接近的初始猜测，帮助数值法收敛
    ik_res = excavator.inverse_kinematics(target_pos, init_guess=[0, 0.1, 0.1, 0.1])
    
    if ik_res:
        print(f"IK 结果关节角: {np.round(ik_res, 3)}")
        # 验证回去
        check_fk = excavator.forward_kinematics(ik_res)
        print(f"IK 结果回代 FK: {np.round(check_fk, 3)}")
        error = np.linalg.norm(np.array(target_pos) - np.array(check_fk))
        print(f"误差: {error:.5f}")
    else:
        print("IK 未找到解 (这是正常的，因为4自由度机械臂难以完全匹配6自由度目标姿态)")