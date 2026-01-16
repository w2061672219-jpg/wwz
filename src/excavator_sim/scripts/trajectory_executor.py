#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹执行节点
从YAML文件读取轨迹并执行
"""

import rospy
from std_msgs.msg import Float64
import yaml
import time

class TrajectoryExecutor:
    def __init__(self):
        rospy.init_node('trajectory_executor', anonymous=True)
        
        # 创建发布者
        self.publishers = {
            'boom': rospy.Publisher('/excavator/boom_position_controller/command', Float64, queue_size=10),
            'arm': rospy.Publisher('/excavator/arm_position_controller/command', Float64, queue_size=10),
            'bucket': rospy.Publisher('/excavator/bucket_position_controller/command', Float64, queue_size=10),
            'swing': rospy.Publisher('/excavator/swing_position_controller/command', Float64, queue_size=10)
        }
        
        # 等待发布者连接
        rospy.sleep(1.0)
    
    def load_trajectory(self, filename):
        """从YAML文件加载轨迹"""
        try:
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
            return data.get('trajectory', {}).get('points', [])
        except Exception as e:
            rospy.logerr(f"加载轨迹文件失败: {e}")
            return []
    
    def execute_trajectory(self, points):
        """执行轨迹"""
        if not points:
            rospy.logwarn("没有轨迹点可执行")
            return
        
        rospy.loginfo(f"开始执行轨迹，共 {len(points)} 个点")
        
        for i, point in enumerate(points):
            if rospy.is_shutdown():
                break
            
            positions = point.get('positions', {})
            duration = point.get('time_from_start', 1.0)
            
            rospy.loginfo(f"执行第 {i+1}/{len(points)} 个点，持续 {duration} 秒")
            
            # 发布各关节位置
            if 'boom' in positions:
                self.publishers['boom'].publish(Float64(positions['boom']))
            if 'arm' in positions:
                self.publishers['arm'].publish(Float64(positions['arm']))
            if 'bucket' in positions:
                self.publishers['bucket'].publish(Float64(positions['bucket']))
            if 'swing' in positions:
                self.publishers['swing'].publish(Float64(positions['swing']))
            
            # 等待
            rospy.sleep(duration)
        
        rospy.loginfo("轨迹执行完成！")
    
    def run(self):
        # 获取轨迹文件路径
        import rospkg
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('excavator_sim')
        trajectory_file = f"{pkg_path}/config/trajectory_fuxi.yaml"
        
        rospy.loginfo(f"加载轨迹文件: {trajectory_file}")
        
        points = self.load_trajectory(trajectory_file)
        
        if points:
            self.execute_trajectory(points)
        else:
            rospy.logerr("无法加载轨迹")

if __name__ == '__main__':
    try:
        executor = TrajectoryExecutor()
        executor.run()
    except rospy.ROSInterruptException:
        pass