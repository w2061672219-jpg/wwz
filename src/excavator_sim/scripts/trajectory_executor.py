#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import time
import numpy as np
import tf2_ros
from std_msgs.msg import Float64

try:
    from kinematics import ExcavatorKinematics
    from plot import plot_trajectories_2d
except ImportError:
    rospy.logerr("错误：请确保 kinematics.py 和 plot.py 在同一目录下！")
    exit()

class FuxiWriter(object):
    def __init__(self):
        rospy.init_node("write_fuxi_trajectory")
        
        self.rate_hz = 10
        self.rate = rospy.Rate(self.rate_hz)
        self.target_pitch = -0.6 

        self.waypoints = self.build_fuxi_waypoints()
        self.kin = ExcavatorKinematics()

        self.pub_swing = rospy.Publisher("/excavator/swing_position_controller/command", Float64, queue_size=10)
        self.pub_boom = rospy.Publisher("/excavator/boom_position_controller/command", Float64, queue_size=10)
        self.pub_arm = rospy.Publisher("/excavator/arm_position_controller/command", Float64, queue_size=10)
        self.pub_bucket = rospy.Publisher("/excavator/bucket_position_controller/command", Float64, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.ref_traj_xyz = []
        self.actual_traj_xyz = []

    def build_fuxi_waypoints(self):
        """
        【重新设计】生成"伏羲"二字的笔画路径点
        
        坐标系说明：
        - X轴：前后方向，X越大越远离挖掘机（向前）
        - Y轴：左右方向，Y正为左，Y负为右
        - Z轴：上下方向
        
        铲斗初始位置约(6.1, 0, 1.15)，所以字要写在X=5.0~6.0范围内
        
        "伏"字在右边（Y为负），"羲"字在左边（Y为正）
        """
        points = []
        
        # === 书写参数设置 ===
        Z_DRAW = 0.5   # 落笔高度
        Z_SAFE = 1.2   # 抬笔高度
        STEP = 0.05    # 插值步长
        
        def add_stroke(start_pt, end_pt):
            """添加一个笔画"""
            # 1. 抬笔移动到起点上方
            points.append([start_pt[0], start_pt[1], Z_SAFE, False])
            # 2. 落笔
            points.append([start_pt[0], start_pt[1], Z_DRAW, True])
            # 3. 插值画线
            dist = math.sqrt((end_pt[0]-start_pt[0])**2 + (end_pt[1]-start_pt[1])**2)
            steps = max(int(dist / STEP), 2)
            for i in range(1, steps + 1):
                alpha = i / float(steps)
                x = start_pt[0] + (end_pt[0] - start_pt[0]) * alpha
                y = start_pt[1] + (end_pt[1] - start_pt[1]) * alpha
                points.append([x, y, Z_DRAW, True])
            # 4. 抬笔
            points.append([end_pt[0], end_pt[1], Z_SAFE, False])

        # ==========================================
        # "伏"字 (右侧，Y为负，中心约 Y=-0.6)
        # 字体范围：X: 5.2~6.0, Y: -1.0~-0.2
        # ==========================================
        
        # --- 单人旁 ---
        # 1. 撇：从右上到左下（X减小，Y略变）
        add_stroke((5.9, -0.3), (5.5, -0.4))
        # 2. 竖：从上到下（X减小，Y不变）
        add_stroke((5.7, -0.35), (5.3, -0.35))
        
        # --- 右边"犬" ---
        # 3. 横：左右方向（X不变，Y变化）
        add_stroke((5.8, -0.5), (5.8, -0.9))
        # 4. 撇：从中间向左下
        add_stroke((5.7, -0.7), (5.3, -0.5))
        # 5. 捺：从中间向右下
        add_stroke((5.6, -0.7), (5.2, -1.0))
        # 6. 点：右上角小点
        add_stroke((5.85, -0.85), (5.8, -0.9))

        # ==========================================
        # "羲"字 (左侧，Y为正，中心约 Y=0.6)
        # 字体范围：X: 5.0~6.0, Y: 0.2~1.0
        # 羲字复杂，简化表示
        # ==========================================

        # --- 上部 "羊" 的简化 ---
        # 两点
        add_stroke((5.95, 0.5), (5.9, 0.55))
        add_stroke((5.95, 0.7), (5.9, 0.75))
        # 三横
        add_stroke((5.85, 0.4), (5.85, 0.9))
        add_stroke((5.75, 0.35), (5.75, 0.95))
        add_stroke((5.65, 0.4), (5.65, 0.9))
        # 中间竖
        add_stroke((5.9, 0.65), (5.55, 0.65))

        # --- 中下部结构简化 ---
        # 横折
        add_stroke((5.5, 0.45), (5.5, 0.85))
        add_stroke((5.5, 0.85), (5.35, 0.85))
        
        # 左下撇
        add_stroke((5.45, 0.5), (5.2, 0.35))
        
        # --- 斜钩（戈的主笔）---
        add_stroke((5.8, 0.3), (5.15, 0.95))
        
        # 右下点
        add_stroke((5.2, 0.9), (5.15, 0.95))

        return points
    
    def get_bucket_tip_position(self):
        """从TF获取铲斗尖端实际位置"""
        try:
            trans = self.tf_buffer.lookup_transform('base_footprint', 'bucket_tip', rospy.Time(0))
            return (trans.transform.translation.x, 
                    trans.transform.translation.y, 
                    trans.transform.translation.z)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
    
    def send_joint_command(self, joints):
        """发送关节指令"""
        swing_rad, boom_rad, arm_rad, bucket_rad = joints
        rospy.loginfo(f"Publishing: swing={math.degrees(swing_rad):.1f}°")
        self.pub_swing.publish(swing_rad)
        self.pub_boom.publish(boom_rad)
        self.pub_arm.publish(arm_rad)
        self.pub_bucket.publish(bucket_rad)
    
    def move_to_position(self, target_pos, wait_time=3.0):
        """
        移动到指定位置并等待到位
        target_pos: [x, y, z]
        wait_time: 等待时间（秒）
        """
        joints = self.kin.inverse_kinematics(target_pos, self.target_pitch)
        if joints:
            self.send_joint_command(joints)
            rospy.loginfo(f"移动到位置: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
            rospy.sleep(wait_time)
            return True
        else:
            rospy.logwarn(f"IK解算失败: {target_pos}")
            return False
    
    def run(self):
        rospy.sleep(1.0)
        rospy.loginfo("开始执行 '伏羲' 书写任务...")
        rospy.loginfo(f"总计路径点: {len(self.waypoints)}")

        if not self.waypoints:
            rospy.logerr("路径点为空，无法执行！")
            return

        # ==================== 关键修正：先移动到起点 ====================
        rospy.loginfo("=" * 50)
        rospy.loginfo("第一步：移动到轨迹起始点上方...")
        rospy.loginfo("=" * 50)
        
        first_pt = self.waypoints[0]
        # 先移动到一个安全的中间位置（在起点上方）
        start_safe_pos = [first_pt[0], first_pt[1], 1.5]
        
        if not self.move_to_position(start_safe_pos, wait_time=8.0):
            rospy.logerr("无法移动到起始位置，退出！")
            return
        
        rospy.loginfo("已到达起始位置上方，开始书写...")
        rospy.loginfo("=" * 50)
        # ==============================================================

        start_time = time.time()
        
        for i, pt in enumerate(self.waypoints):
            if rospy.is_shutdown(): 
                break
            
            target_x, target_y, target_z, is_drawing = pt
            
            joints = self.kin.inverse_kinematics([target_x, target_y, target_z], self.target_pitch)
            
            if joints:
                self.send_joint_command(joints)
                
                # 记录数据（仅在落笔时）
                if is_drawing:
                    self.ref_traj_xyz.append([target_x, target_y, target_z])
                    actual_pos = self.get_bucket_tip_position()
                    if actual_pos is not None:
                        self.actual_traj_xyz.append(actual_pos)
                
                # 进度显示
                if i % 20 == 0:
                    rospy.loginfo(f"进度: {i}/{len(self.waypoints)} ({100*i/len(self.waypoints):.1f}%)")
            else:
                rospy.logwarn(f"点 {i} IK解算失败: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")

            self.rate.sleep()

        duration = time.time() - start_time
        rospy.loginfo(f"关节指令发送完毕! 耗时: {duration:.2f}秒")
        
        # === 等待挖掘机完成运动 ===
        rospy.loginfo("等待挖掘机完成最后的运动（10秒）...")
        
        # 持续采集实际轨迹
        for i in range(100):  # 10秒
            if rospy.is_shutdown():
                break
            actual_pos = self.get_bucket_tip_position()
            if actual_pos is not None and len(self.actual_traj_xyz) > 0:
                # 可以选择追加或更新
                pass
            rospy.sleep(0.1)
        
        # === 绘图 ===
        rospy.loginfo("=" * 50)
        rospy.loginfo("正在生成 2D 轨迹对比图...")
        rospy.loginfo(f"期望轨迹点数: {len(self.ref_traj_xyz)}")
        rospy.loginfo(f"实际轨迹点数: {len(self.actual_traj_xyz)}")
        
        if len(self.ref_traj_xyz) > 0:
            plot_trajectories_2d(self.ref_traj_xyz, self.actual_traj_xyz, save_path="fuxi_2d_final.png")
            rospy.loginfo("图片已保存: fuxi_2d_final.png")
        else:
            rospy.logerr("没有记录到轨迹数据！")

if __name__ == "__main__":
    try:
        writer = FuxiWriter()
        writer.run()
    except rospy.ROSInterruptException:
        pass