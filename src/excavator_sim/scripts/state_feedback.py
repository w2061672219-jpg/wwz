#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
挖掘机状态反馈节点
订阅关节状态并显示
"""

import rospy
from sensor_msgs.msg import JointState
import math

class StateFeedback:
    def __init__(self):
        rospy.init_node('state_feedback', anonymous=True)
        
        # 订阅关节状态
        self.sub = rospy.Subscriber(
            '/excavator/joint_states',
            JointState,
            self.joint_state_callback)
        
        self.joint_data = {}
        
    def joint_state_callback(self, msg):
        """处理关节状态消息"""
        for i, name in enumerate(msg.name):
            self.joint_data[name] = {
                'position': msg.position[i] if i < len(msg.position) else 0,
                'velocity': msg.velocity[i] if i < len(msg.velocity) else 0,
                'effort': msg.effort[i] if i < len(msg.effort) else 0
            }
    
    def print_state(self):
        """打印当前状态"""
        print("\n" + "="*60)
        print("挖掘机关节状态")
        print("="*60)
        print(f"{'关节名称':<20} {'位置(rad)':<12} {'速度':<12} {'力矩':<12}")
        print("-"*60)
        
        for name, data in self.joint_data.items():
            pos_deg = math.degrees(data['position'])
            print(f"{name:<20} {data['position']:>+8.4f} ({pos_deg:>+6.1f}°) "
                  f"{data['velocity']:>+8.4f} {data['effort']:>+8.2f}")
    
    def run(self):
        rate = rospy.Rate(2)  # 2Hz 更新显示
        
        while not rospy.is_shutdown():
            if self.joint_data:
                self.print_state()
            rate.sleep()

if __name__ == '__main__':
    try:
        feedback = StateFeedback()
        feedback.run()
    except rospy.ROSInterruptException:
        pass