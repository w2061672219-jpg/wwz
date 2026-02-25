#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import time
import numpy as np
import PyKDL
import matplotlib.pyplot as plt
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

# =========================================
# 1. 运动学类 (包含智能搜索 IK)
# =========================================
class ExcavatorKinematicsKDL:
    def __init__(self):
        # 几何参数 (基于 URDF)
        self.H_BASE_TOTAL = 1.65
        self.OFFSET_BOOM_X = 0.8 
        self.L_BOOM = 2.5
        self.L_ARM = 2.0
        self.L_BUCKET_X = 0.8 
        self.L_BUCKET_Z = -0.5

        # 构建 KDL 链条
        self.chain = PyKDL.Chain()
        self.chain.addSegment(PyKDL.Segment(PyKDL.Joint(PyKDL.Joint.RotY), PyKDL.Frame(PyKDL.Vector(self.L_BOOM, 0, 0))))
        self.chain.addSegment(PyKDL.Segment(PyKDL.Joint(PyKDL.Joint.RotY), PyKDL.Frame(PyKDL.Vector(self.L_ARM, 0, 0))))
        vec_tip = PyKDL.Vector(self.L_BUCKET_X, 0, self.L_BUCKET_Z)
        self.chain.addSegment(PyKDL.Segment(PyKDL.Joint(PyKDL.Joint.RotY), PyKDL.Frame(vec_tip)))

        # 求解器
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.ik_solver_vel = PyKDL.ChainIkSolverVel_pinv(self.chain)
        
        # 关节限制 (宽限版)
        self.q_min = PyKDL.JntArray(3)
        self.q_max = PyKDL.JntArray(3)
        self.q_min[0] = -2.0; self.q_max[0] = 1.5   # Boom
        self.q_min[1] = -2.8; self.q_max[1] = 1.0   # Arm
        self.q_min[2] = -3.0; self.q_max[2] = 3.0   # Bucket (非常宽)

        self.ik_solver = PyKDL.ChainIkSolverPos_NR_JL(
            self.chain, self.q_min, self.q_max, 
            self.fk_solver, self.ik_solver_vel, 200, 1e-4
        )
        
        # 缓存上一次的解作为 Seed，防止抽搐
        self.last_q = PyKDL.JntArray(3)
        self.last_q[0] = -0.5
        self.last_q[1] = -1.0
        self.last_q[2] = -0.5

    def solve_ik_smart(self, x_global, z_global, preferred_pitch=-1.57):
        """
        智能 IK：如果首选角度不可达，尝试搜索附近的角度。
        优先保证 (x, z) 到达。
        """
        # 1. 坐标转换：全局 -> 局部
        x_local = x_global - self.OFFSET_BOOM_X
        z_local = z_global - self.H_BASE_TOTAL
        
        target_pos = PyKDL.Vector(x_local, 0, z_local)
        
        # 2. 定义搜索范围：优先 Pitch, 然后向两侧搜索
        # 搜索范围：首选角度 +/- 1.0 弧度 (约 +/- 57度)
        search_steps = [0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.5, -0.5, 0.8, -0.8]
        
        for delta in search_steps:
            test_pitch = preferred_pitch + delta
            
            # 构建目标 Frame
            target_rot = PyKDL.Rotation.RotY(test_pitch)
            target_frame = PyKDL.Frame(target_rot, target_pos)
            
            q_out = PyKDL.JntArray(3)
            ret = self.ik_solver.CartToJnt(self.last_q, target_frame, q_out)
            
            if ret >= 0:
                # 找到解了！更新 last_q 并返回
                self.last_q = q_out
                # 转换回 Python 列表
                return [q_out[0], q_out[1], q_out[2]], test_pitch
        
        # 如果所有角度都尝试失败
        return None, None

    def get_current_fk(self, q_list):
        # 正运动学，用于绘图
        q_in = PyKDL.JntArray(3)
        for i in range(3): q_in[i] = q_list[i]
        
        end_frame = PyKDL.Frame()
        self.fk_solver.JntToCart(q_in, end_frame)
        
        x_local = end_frame.p[0]
        z_local = end_frame.p[2]
        
        x_global = x_local + self.OFFSET_BOOM_X
        z_global = z_local + self.H_BASE_TOTAL
        return x_global, z_global

# =========================================
# 2. 执行与绘图逻辑
# =========================================
class SmartWriter:
    def __init__(self):
        rospy.init_node("smart_writer")
        self.kin = ExcavatorKinematicsKDL()
        
        # 发布器
        self.pub_swing = rospy.Publisher("/excavator/swing_position_controller/command", Float64, queue_size=10)
        self.pub_boom = rospy.Publisher("/excavator/boom_position_controller/command", Float64, queue_size=10)
        self.pub_arm = rospy.Publisher("/excavator/arm_position_controller/command", Float64, queue_size=10)
        self.pub_bucket = rospy.Publisher("/excavator/bucket_position_controller/command", Float64, queue_size=10)
        
        # 记录数据
        self.ref_traj = []   # [x, y]
        self.actual_traj = [] # [x, y]
        
        # 当前关节状态（用于绘图反馈）
        self.current_joints = [0,0,0,0] # swing, boom, arm, bucket
        rospy.Subscriber("/excavator/joint_states", JointState, self.cb_joints)
        
        time.sleep(1.0) # 等待连接

    def cb_joints(self, msg):
        # 简单的映射，你需要根据实际 joint_states 的 name 顺序调整
        # 假设顺序是 swing, boom, arm, bucket
        try:
            # 这里简化处理，实际请通过 name 匹配
            self.current_joints = msg.position 
        except:
            pass

    def run_square_test(self):
        """画一个更合理位置的正方形"""
        # 中心点设置在 (4.0, 0, 0.5)
        center_x = 4.0
        center_y = 0.0
        draw_z = 0.5
        size = 0.8  # 边长
        
        # 生成路径点 (密集插值)
        waypoints = []
        
        # 定义四个角 (右下 -> 右上 -> 左上 -> 左下 -> 右下)
        # 注意：这里 Y 正是左，Y 负是右
        corners = [
            (center_x - size/2, center_y - size/2), # 近右
            (center_x + size/2, center_y - size/2), # 远右
            (center_x + size/2, center_y + size/2), # 远左
            (center_x - size/2, center_y + size/2), # 近左
            (center_x - size/2, center_y - size/2)  # 回到起点
        ]
        
        # 插值生成
        resolution = 0.02 # 每 2cm 一个点
        for i in range(len(corners)-1):
            p1 = corners[i]
            p2 = corners[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            steps = int(dist / resolution)
            for s in range(steps):
                alpha = s / float(steps)
                x = p1[0] + (p2[0] - p1[0]) * alpha
                y = p1[1] + (p2[1] - p1[1]) * alpha
                waypoints.append([x, y, draw_z])
                
        rospy.loginfo(f"生成了 {len(waypoints)} 个轨迹点，开始执行...")
        
        rate = rospy.Rate(20) # 20Hz
        
        for pt in waypoints:
            target_x, target_y, target_z = pt
            
            # 1. Swing 解算 (解析解)
            # swing 角度 = atan2(y, x)
            # 注意：实际距离要用 xy 平面投影距离
            r_ground = math.hypot(target_x, target_y)
            q_swing = math.atan2(target_y, target_x)
            
            # 2. 机械臂平面距离
            # 机械臂只需要伸到 r_ground 这个距离，不管 y 是多少
            x_planar = r_ground
            
            # 3. 智能 IK 解算
            # 尝试垂直向下 (Pitch = -pi/2 = -1.57) 附近寻找解
            q_list, final_pitch = self.kin.solve_ik_smart(x_planar, target_z, preferred_pitch=-1.57)
            
            if q_list:
                # 发布命令
                self.pub_swing.publish(q_swing)
                self.pub_boom.publish(q_list[0])
                self.pub_arm.publish(q_list[1])
                self.pub_bucket.publish(q_list[2])
                
                # 记录期望轨迹
                self.ref_traj.append([target_x, target_y])
                
                # 计算并记录实际轨迹 (利用 FK)
                # 注意：这里需要把 Swing 加回来算实际坐标
                # 简化：假设 swing 完美跟随，只算平面 FK 然后旋转
                fk_x_planar, fk_z = self.kin.get_current_fk([q_list[0], q_list[1], q_list[2]])
                real_x = fk_x_planar * math.cos(q_swing)
                real_y = fk_x_planar * math.sin(q_swing)
                self.actual_traj.append([real_x, real_y])
                
                # 如果 Pitch 调整很大，打印警告
                if abs(final_pitch - (-1.57)) > 0.5:
                    rospy.logwarn(f"调整姿态以触达点: Pitch {final_pitch:.2f}")
            else:
                rospy.logerr(f"点不可达: [{target_x:.2f}, {target_z:.2f}]")
            
            rate.sleep()
            
        self.plot_result()

    def plot_result(self):
        rospy.loginfo("正在绘图...")
        ref = np.array(self.ref_traj)
        act = np.array(self.actual_traj)
        
        plt.figure(figsize=(10, 10))
        if len(ref) > 0:
            # 翻转 Y 轴以符合直觉 (左正右负 -> 画图时左边在图左边)
            plt.plot(-ref[:, 1], ref[:, 0], 'r--', label='Reference', linewidth=2)
        if len(act) > 0:
            plt.plot(-act[:, 1], act[:, 0], 'b-', label='Actual (FK)', linewidth=1)
            
        plt.title("Excavator Smart IK Test")
        plt.xlabel("Left <--- Y (m) ---> Right")
        plt.ylabel("Front X (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig("smart_square_result.png")
        plt.show()

if __name__ == '__main__':
    try:
        writer = SmartWriter()
        writer.run_square_test()
    except rospy.ROSInterruptException:
        pass