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
# 1. 挖掘机运动学类 (修复了坐标转换 Bug)
# =========================================
class ExcavatorKinematics:
    def __init__(self):
        # --- 几何参数 ---
        # 垂直高度: 地面(0) -> 底盘 -> 回转中心 -> 大臂根部
        self.H_BOOM_ROOT_GLOBAL = 1.65 
        # 水平偏移: 底盘中心 -> 大臂根部 (Swing X + Boom Offset X)
        self.OFFSET_BOOM_X_GLOBAL = 0.8

        # 连杆长度
        self.L_BOOM = 2.5
        self.L_ARM = 2.0
        self.L_BUCKET_X = 0.8
        self.L_BUCKET_Z = -0.5

        # --- KDL Chain (局部坐标系: Boom Root -> Tip) ---
        self.chain = PyKDL.Chain()
        self.chain.addSegment(PyKDL.Segment(PyKDL.Joint(PyKDL.Joint.RotY), PyKDL.Frame(PyKDL.Vector(self.L_BOOM, 0, 0))))
        self.chain.addSegment(PyKDL.Segment(PyKDL.Joint(PyKDL.Joint.RotY), PyKDL.Frame(PyKDL.Vector(self.L_ARM, 0, 0))))
        vec_tip = PyKDL.Vector(self.L_BUCKET_X, 0, self.L_BUCKET_Z)
        self.chain.addSegment(PyKDL.Segment(PyKDL.Joint(PyKDL.Joint.RotY), PyKDL.Frame(vec_tip)))

        # 求解器
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.ik_solver_pos = PyKDL.ChainIkSolverPos_LMA(self.chain) # Levenberg-Marquardt 求解器

        # 关节限制 (用于 IK 约束)
        self.q_min = PyKDL.JntArray(3); self.q_max = PyKDL.JntArray(3)
        self.q_min[0] = -2.0; self.q_max[0] = 1.0  # Boom
        self.q_min[1] = -2.8; self.q_max[1] = 1.0  # Arm
        self.q_min[2] = -3.5; self.q_max[2] = 3.5  # Bucket

        # 记录上一次的解作为种子，防止 IK 跳变
        self.last_q = PyKDL.JntArray(3)

    def global_to_local(self, x_global, y_global, z_global):
        """
        将全局坐标 (相对于 base_footprint) 转换为 局部坐标 (相对于 Boom Root 平面)
        """
        # 1. 计算回转角 (Swing)
        theta_swing = math.atan2(y_global, x_global)
        
        # 2. 计算平面内的水平距离 (Radius)
        # 这里的 r_global 是从底盘中心到目标的水平距离
        r_global = math.sqrt(x_global**2 + y_global**2)
        
        # 3. 转换为局部 Radius (扣除大臂根部的水平偏移)
        # 注意：大臂根部通常在回转中心前方 0.8m 处
        r_local = r_global - self.OFFSET_BOOM_X_GLOBAL
        
        # 4. 转换为局部 Height (扣除大臂根部的垂直高度)
        z_local = z_global - self.H_BOOM_ROOT_GLOBAL
        
        return theta_swing, r_local, z_local

    def solve_ik_smart(self, x, y, z, preferred_pitch=-0.8):
        """
        智能 IK: 自动搜索可行的铲斗角度
        """
        # 1. 坐标转换
        swing_angle, r_local, z_local = self.global_to_local(x, y, z)

        # 2. 尝试求解
        # 我们在 preferred_pitch 附近搜索可行解
        search_range = np.linspace(0, 2.0, 20) # 向两个方向搜索
        pitch_candidates = [preferred_pitch]
        for delta in search_range:
            pitch_candidates.append(preferred_pitch + delta)
            pitch_candidates.append(preferred_pitch - delta)

        for pitch in pitch_candidates:
            # 构建目标帧 (在局部平面内)
            # 目标位置 (r_local, 0, z_local)
            pos_vec = PyKDL.Vector(r_local, 0, z_local)
            # 目标旋转 (绕 Y 轴转 pitch)
            rot = PyKDL.Rotation.RotY(pitch)
            target_frame = PyKDL.Frame(rot, pos_vec)

            q_out = PyKDL.JntArray(3)
            # 使用上一次的解作为种子
            ret = self.ik_solver_pos.CartToJnt(self.last_q, target_frame, q_out)

            if ret >= 0:
                # 检查是否在关节限制内
                if (self.q_min[0] < q_out[0] < self.q_max[0] and
                    self.q_min[1] < q_out[1] < self.q_max[1]):
                    
                    self.last_q = q_out # 更新种子
                    return True, swing_angle, q_out[0], q_out[1], q_out[2]

        return False, 0, 0, 0, 0

    def get_fk_position(self, boom_q, arm_q, bucket_q, swing_q):
        """正运动学：从关节角反推末端坐标 (用于画实际轨迹)"""
        q_in = PyKDL.JntArray(3)
        q_in[0] = boom_q; q_in[1] = arm_q; q_in[2] = bucket_q
        
        frame_out = PyKDL.Frame()
        self.fk_solver.JntToCart(q_in, frame_out)
        
        # 局部 -> 全局
        r_local = frame_out.p.x()
        z_local = frame_out.p.z()
        
        r_global = r_local + self.OFFSET_BOOM_X_GLOBAL
        x = r_global * math.cos(swing_q)
        y = r_global * math.sin(swing_q)
        z = z_local + self.H_BOOM_ROOT_GLOBAL
        return x, y, z

# =========================================
# 2. 书写控制器
# =========================================
class FuxiWriter:
    def __init__(self):
        rospy.init_node("write_fu_final")
        self.kin = ExcavatorKinematics()
        
        # ROS Publishers
        self.pub_swing = rospy.Publisher("/excavator/swing_position_controller/command", Float64, queue_size=10)
        self.pub_boom = rospy.Publisher("/excavator/boom_position_controller/command", Float64, queue_size=10)
        self.pub_arm = rospy.Publisher("/excavator/arm_position_controller/command", Float64, queue_size=10)
        self.pub_bucket = rospy.Publisher("/excavator/bucket_position_controller/command", Float64, queue_size=10)

        # 记录实际关节状态
        self.current_joints = {'swing': 0, 'boom': 0, 'arm': 0, 'bucket': 0}
        rospy.Subscriber("/excavator/joint_states", JointState, self.joint_cb)

        # 绘图数据
        self.ref_strokes = []    # 期望笔画 [[(x,y), (x,y)...], [...]]
        self.actual_strokes = [] # 实际笔画
        self.current_ref_stroke = []
        self.current_act_stroke = []
        self.is_drawing = False

        time.sleep(1.0) # 等待连接

    def joint_cb(self, msg):
        # 更鲁棒的映射：尝试通过名字子串匹配不同命名风格，并输出调试信息
        try:
            names = list(msg.name)
            pos = list(msg.position)

            # 对每个期望的关节名，尝试通过子串匹配对应的索引
            mapped = {}
            for i, n in enumerate(names):
                ln = n.lower()
                if 'swing' in ln:
                    self.current_joints['swing'] = pos[i]
                    mapped['swing'] = True
                if 'boom' in ln:
                    self.current_joints['boom'] = pos[i]
                    mapped['boom'] = True
                if 'arm' in ln:
                    # 直接匹配包含 'arm'
                    self.current_joints['arm'] = pos[i]
                    mapped['arm'] = True
                if 'bucket' in ln:
                    self.current_joints['bucket'] = pos[i]
                    mapped['bucket'] = True

            missing = [k for k in ['swing', 'boom', 'arm', 'bucket'] if k not in mapped]
            if missing:
                rospy.loginfo(f"joint_cb: missing joints {missing}; available names={names}")
            else:
                rospy.loginfo(f"joint_cb: updated joints {self.current_joints}")
        except Exception as e:
            rospy.loginfo(f"joint_cb exception: {e}")

    def run(self):
        # 1. 生成“伏”字轨迹 (带抬笔/落笔标志)
        # 格式: [x, y, z, pen_down]
        waypoints = self.generate_fu_character()
        
        print(f"生成轨迹点数: {len(waypoints)}")
        
        rate = rospy.Rate(50) # 50Hz 控制频率
        
        for wp in waypoints:
            target_x, target_y, target_z, pen_down = wp
            
            # IK 解算
            success, swing, boom, arm, bucket = self.kin.solve_ik_smart(target_x, target_y, target_z)
            
            if success:
                # 发送命令
                self.pub_swing.publish(swing)
                self.pub_boom.publish(boom)
                self.pub_arm.publish(arm)
                self.pub_bucket.publish(bucket)

                # --- 绘图数据记录 ---
                if pen_down:
                    if not self.is_drawing:
                        # 刚开始落笔，开启新的一笔
                        self.is_drawing = True
                        self.current_ref_stroke = []
                        self.current_act_stroke = []
                        self.ref_strokes.append(self.current_ref_stroke)
                        self.actual_strokes.append(self.current_act_stroke)
                    
                    # 记录期望点
                    self.current_ref_stroke.append([target_x, target_y])
                    
                    # 计算并记录实际点 (FK)
                    # 等待短暂时间让控制生效（如果有订阅到 joint_states，会在回调中更新）
                    rospy.sleep(0.02)

                    # 如果当前没有及时反馈关节状态，为了能看到轨迹，我们使用发布的命令角度作为近似实际值
                    try:
                        act_x, act_y, act_z = self.kin.get_fk_position(
                            boom, arm, bucket, swing
                        )
                    except Exception:
                        # 备用：若上述失败，使用最新的已接收关节状态
                        act_x, act_y, act_z = self.kin.get_fk_position(
                            self.current_joints['boom'], 
                            self.current_joints['arm'], 
                            self.current_joints['bucket'], 
                            self.current_joints['swing']
                        )
                    self.current_act_stroke.append([act_x, act_y])
                else:
                    self.is_drawing = False
            else:
                rospy.logwarn(f"不可达: {target_x:.2f}, {target_y:.2f}, {target_z:.2f}")

            rate.sleep()

        # 书写结束，画图
        self.plot_result()

    def plot_result(self):
        plt.figure(figsize=(10, 10))
        plt.title("Excavator Calligraphy: 'Fu' (Reference vs Actual)")
        plt.xlabel("Left <--- Y Axis (m) ---> Right")
        plt.ylabel("Forward (X Axis) (m)")
        plt.axis('equal')
        plt.grid(True)

        # 绘制期望轨迹 (红色虚线)
        first_ref = True
        for stroke in self.ref_strokes:
            arr = np.array(stroke)
            if len(arr) > 0:
                # 注意：Plot X轴设为 Robot Y, Plot Y轴设为 Robot X
                lbl = 'Reference' if first_ref else None
                plt.plot(arr[:, 1], arr[:, 0], 'r--', linewidth=2, label=lbl, alpha=0.7)
                first_ref = False

        # 绘制实际轨迹 (蓝色实线)
        first_act = True
        for stroke in self.actual_strokes:
            arr = np.array(stroke)
            if len(arr) > 0:
                lbl = 'Actual' if first_act else None
                plt.plot(arr[:, 1], arr[:, 0], 'b-', linewidth=1.5, label=lbl)
                first_act = False

        plt.legend()
        plt.savefig("fu_final_result.png")
        print("绘图完成，已保存为 fu_final_result.png")
        plt.show()

    def generate_fu_character(self):
        """生成“伏”字的密集轨迹点"""
        pts = []
        Z_UP = 1.0  # 抬笔高度
        Z_DOWN = 0.5 # 落笔高度 (地面偏移)
        CENTER_X = 4.5
        SCALE = 0.6 # 字的大小缩放

        def add_stroke(start, end, steps=30):
            # 1. 移动到起点上方 (快速)
            pts.append([start[0], start[1], Z_UP, False])
            # 2. 落笔 (慢)
            pts.append([start[0], start[1], Z_DOWN, True])
            # 3. 直线插值
            for i in range(steps + 1):
                alpha = i / float(steps)
                x = start[0] + (end[0] - start[0]) * alpha
                y = start[1] + (end[1] - start[1]) * alpha
                pts.append([x, y, Z_DOWN, True])
            # 4. 抬笔
            pts.append([end[0], end[1], Z_UP, False])

        # === 构造“伏”字笔画 (坐标：X前后，Y左右) ===
        # 左边“亻” (单人旁) -> Y 偏正 (左侧)
        # 1. 撇 (Top to Bottom-Left)
        add_stroke([CENTER_X + 0.5*SCALE, 0.4*SCALE], [CENTER_X - 0.2*SCALE, 0.8*SCALE])
        # 2. 竖 (Middle to Bottom)
        add_stroke([CENTER_X + 0.1*SCALE, 0.6*SCALE], [CENTER_X - 0.8*SCALE, 0.6*SCALE])

        # 右边“犬” -> Y 偏负 (右侧)
        # 3. 横 (Horizontal)
        add_stroke([CENTER_X + 0.1*SCALE, -0.2*SCALE], [CENTER_X + 0.2*SCALE, -1.0*SCALE])
        # 4. 撇 (Curve down-left) -> 简化为直线
        add_stroke([CENTER_X + 0.5*SCALE, -0.6*SCALE], [CENTER_X - 0.8*SCALE, -0.2*SCALE])
        # 5. 捺 (Down-Right)
        add_stroke([CENTER_X - 0.1*SCALE, -0.6*SCALE], [CENTER_X - 0.8*SCALE, -1.2*SCALE])
        # 6. 点 (Top Right)
        add_stroke([CENTER_X + 0.4*SCALE, -0.3*SCALE], [CENTER_X + 0.5*SCALE, -0.2*SCALE])

        return pts

if __name__ == '__main__':
    try:
        writer = FuxiWriter()
        writer.run()
    except rospy.ROSInterruptException:
        pass