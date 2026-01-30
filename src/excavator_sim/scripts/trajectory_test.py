#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import time
import numpy as np
import tf2_ros  # 引入 TF2
from std_msgs.msg import Float64
from geometry_msgs.msg import TransformStamped

# 尝试导入绘图和运动学库
try:
    from kinematics import ExcavatorKinematics
    from plot import plot_trajectories_2d
except ImportError:
    rospy.logerr("错误：请确保 kinematics.py 和 plot.py 在同一目录下！")
    exit()

class FuxiWriter(object):
    def __init__(self):
        # 1. 初始化节点
        rospy.init_node("write_fuxi_realtime")
        
        # 2. 初始化 TF 监听器 (核心修改部分)
        #    这将允许我们查询 "bucket_tip" 在 "base_footprint" 坐标系下的真实位置
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 给 TF 一点时间建立缓存
        rospy.sleep(1.0)

        # 3. 初始化控制频率
        self.rate_hz = 10
        self.rate = rospy.Rate(self.rate_hz)
        self.target_pitch = -0.6 

        # 4. 生成“伏羲”轨迹点 (期望路径)
        self.waypoints = self.build_fuxi_waypoints()
        
        # 5. 运动学解算器 (用于计算发给电机的角度)
        self.kin = ExcavatorKinematics()

        # 6. 初始化 Publishers
        self.pub_swing = rospy.Publisher("/excavator/swing_position_controller/command", Float64, queue_size=10)
        self.pub_boom = rospy.Publisher("/excavator/boom_position_controller/command", Float64, queue_size=10)
        self.pub_arm = rospy.Publisher("/excavator/arm_position_controller/command", Float64, queue_size=10)
        self.pub_bucket = rospy.Publisher("/excavator/bucket_position_controller/command", Float64, queue_size=10)

        # 7. 数据记录
        self.ref_traj_xyz = [] # 记录期望轨迹
        self.real_traj_xyz = []  # 记录通过 TF 读到的真实轨迹

    def build_fuxi_waypoints(self):
        """
        生成“伏羲”二字的笔画路径点 (安全范围版)
        """
        points = []
        # 书写参数
        Z_DRAW = 0.5
        Z_SAFE = 1.2
        STEP = 0.05
        
        def add_stroke(start_pt, end_pt):
            # 抬笔
            points.append([start_pt[0], start_pt[1], Z_SAFE, False])
            # 落笔准备
            points.append([start_pt[0], start_pt[1], Z_DRAW, True])
            # 插值画线
            dist = math.sqrt((end_pt[0]-start_pt[0])**2 + (end_pt[1]-start_pt[1])**2)
            steps = int(dist / STEP)
            if steps < 2: steps = 2
            for i in range(1, steps + 1):
                alpha = i / float(steps)
                x = start_pt[0] + (end_pt[0] - start_pt[0]) * alpha
                y = start_pt[1] + (end_pt[1] - start_pt[1]) * alpha
                points.append([x, y, Z_DRAW, True])
            # 抬笔结束
            points.append([end_pt[0], end_pt[1], Z_SAFE, False])

        # === 伏 (X: 3.5~4.8, Y: 0.2~1.5) ===
        # 单人旁
        add_stroke((4.5, 0.6), (4.3, 0.8)) # 撇
        add_stroke((4.4, 0.7), (4.0, 0.7)) # 竖
        # 右边犬
        add_stroke((4.5, 1.0), (4.5, 1.4)) # 横
        add_stroke((4.5, 1.2), (3.8, 1.0)) # 撇弯
        add_stroke((4.2, 1.1), (3.8, 1.4)) # 捺
        add_stroke((4.4, 1.3), (4.5, 1.5)) # 点

        # === 羲 (X: 3.5~4.8, Y: -1.5~-0.2) ===
        # 羊
        add_stroke((4.7, -0.8), (4.7, -0.4)) # 点/横
        add_stroke((4.6, -1.0), (4.6, -0.2)) # 横
        add_stroke((4.5, -0.6), (4.2, -0.6)) # 竖
        add_stroke((4.4, -1.0), (4.4, -0.2)) # 横
        add_stroke((4.2, -1.2), (4.2, 0.0))  # 长横
        # 秀 (简写结构)
        add_stroke((4.0, -1.0), (3.6, -1.2)) # 撇
        add_stroke((4.0, -0.2), (3.6, 0.0))  # 捺
        add_stroke((3.9, -0.6), (3.5, -0.8)) # 戈钩部分

        return points

    def get_real_pose_from_tf(self):
        """
        查询 TF 树，获取当前真实的末端坐标
        返回: [x, y, z] 或 None
        """
        try:
            # 这里的 'base_footprint' 和 'bucket_tip' 需要和你的 URDF/TF树名称一致
            # 如果报错找不到 bucket_tip，请尝试改成 bucket_link
            trans = self.tf_buffer.lookup_transform('base_footprint', 'bucket_tip', rospy.Time(0))
            
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
            return [x, y, z]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # 偶尔查询失败是正常的，尤其是刚启动时
            return None

    def run(self):
        rospy.loginfo(f"开始执行 '伏羲' 书写任务，总路径点: {len(self.waypoints)}")
        rospy.loginfo("正在通过 TF 监听真实轨迹...")

        start_time = rospy.Time.now()

        for i, pt in enumerate(self.waypoints):
            if rospy.is_shutdown():
                break

            target_x, target_y, target_z, is_draw = pt

            # 1. 逆运动学计算 (IK)
            # 注意：这里我们只计算目标的 joint 角度发给控制器
            q = self.kin.inverse_kinematics([target_x, target_y, target_z], self.target_pitch)
            
            if q is None:
                rospy.logwarn(f"点 {i} IK解算失败: ({target_x:.2f}, {target_y:.2f})")
                continue

            # 2. 发送控制指令
            self.pub_swing.publish(q[0])
            self.pub_boom.publish(q[1])
            self.pub_arm.publish(q[2])
            self.pub_bucket.publish(q[3])

            # 3. 记录期望轨迹 (用于画图中的红线)
            if is_draw:
                self.ref_traj_xyz.append([target_x, target_y, target_z])

            # 4. 等待执行 (物理运动需要时间)
            self.rate.sleep()

            # 5. 【核心】读取并记录真实位置 (用于画图中的蓝线)
            #    在 sleep 之后读取，代表“此刻机器人在哪里”
            real_pos = self.get_real_pose_from_tf()
            if real_pos:
                # 只有当高度接近书写高度时才记录，避免抬笔时的轨迹干扰视觉（可选）
                # 这里为了看清所有动作，我们全部记录，或者只记录 is_draw 的部分
                if is_draw: 
                    self.real_traj_xyz.append(real_pos)

        end_time = rospy.Time.now()
        duration = (end_time - start_time).to_sec()
        rospy.loginfo(f"任务完成! 耗时: {duration:.2f} 秒")
        
        # 6. 绘图保存
        # 注意：这里传入的是 self.real_traj_xyz (TF数据)，不再是 FK 计算值
        rospy.loginfo(f"正在绘图... 期望点数:{len(self.ref_traj_xyz)}, 真实点数:{len(self.real_traj_xyz)}")
        
        if len(self.real_traj_xyz) > 0:
            plot_trajectories_2d(self.ref_traj_xyz, self.real_traj_xyz, "fuxi_realtime_tf.png")
        else:
            rospy.logerr("未采集到 TF 数据，无法绘图。请检查 TF Tree 是否正常 (rosrun rqt_tf_tree rqt_tf_tree)")

if __name__ == "__main__":
    try:
        writer = FuxiWriter()
        writer.run()
    except rospy.ROSInterruptException:
        pass