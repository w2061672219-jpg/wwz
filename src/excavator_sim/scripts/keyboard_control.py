#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
挖掘机键盘控制节点
使用键盘控制各个关节的运动
"""

import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import sys
import termios
import tty

class KeyboardController:
    def __init__(self):
        rospy.init_node('keyboard_controller', anonymous=True)
        
        # 创建各关节的发布者
        self.pub_boom = rospy.Publisher(
            '/excavator/boom_position_controller/command',
            Float64, queue_size=10)
        self.pub_arm = rospy.Publisher(
            '/excavator/arm_position_controller/command',
            Float64, queue_size=10)
        self.pub_bucket = rospy.Publisher(
            '/excavator/bucket_position_controller/command',
            Float64, queue_size=10)
        self.pub_swing = rospy.Publisher(
            '/excavator/swing_position_controller/command',
            Float64, queue_size=10)
        self.pub_cmd_vel = rospy.Publisher(
            '/excavator/cmd_vel',
            Twist, queue_size=10)
        
        # 当前关节位置
        self.boom_pos = 0.0
        self.arm_pos = 0.0
        self.bucket_pos = 0.0
        self.swing_pos = 0.0
        # 当前底盘速度状态
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        
        # 步进量
        self.step = 0.05
        self.max_linear_vel = 2.0  # 最大线速度 m/s
        self.max_angular_vel = 1.0  # 最大角速度 rad/s
        
        self.print_help()
    
    def print_help(self):
        msg = """
╔════════════════════════════════════════════╗
║       挖掘机键盘控制                       ║
╠════════════════════════════════════════════╣
║  W/S : 大臂 上升/下降                      ║
║  A/D : 小臂 伸出/收回                      ║
║  Q/E : 铲斗 上翻/下翻                      ║
║  Z/C : 回转 左转/右转                      ║
║  I   : 前进 (加速)                         ║
║  K   : 后退 (减速/倒车)                    ║
║  J   : 左转                                ║
║  L   : 右转                                ║
║  Space: 底盘急停 (速度置0)                 ║
║  R   : 重置所有关节                        ║
║  +/- : 增加/减少步进量                     ║
║  H   : 显示帮助                            ║
║  Ctrl+C : 退出                             ║
╚════════════════════════════════════════════╝
        """
        print(msg)
        print(f"当前步进量: {self.step:.3f} rad")
    
    def get_key(self):
        """获取键盘输入"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def run(self):
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            key = self.get_key()
            
            if key == '\x03':  # Ctrl+C
                twist = Twist()
                self.pub_cmd_vel.publish(twist)
                break
            
            # 大臂控制
            if key.lower() == 'w':
                self.boom_pos += self.step
            elif key.lower() == 's':
                self.boom_pos -= self.step
            
            # 小臂控制
            elif key.lower() == 'a':
                self.arm_pos += self.step
            elif key.lower() == 'd':
                self.arm_pos -= self.step
            
            # 铲斗控制
            elif key.lower() == 'q':
                self.bucket_pos += self.step
            elif key.lower() == 'e':
                self.bucket_pos -= self.step
            
            # 回转控制
            elif key.lower() == 'z':
                self.swing_pos += self.step
            elif key.lower() == 'c':
                self.swing_pos -= self.step

            # 底盘控制
            elif key.lower() == 'i':
                self.target_linear_vel += self.step
            elif key.lower() == 'k':
                self.target_linear_vel -= self.step
            elif key.lower() == 'j':
                self.target_angular_vel += self.step
            elif key.lower() == 'l':
                self.target_angular_vel -= self.step
            elif key == ' ':  # 空格键急停
                self.target_linear_vel = 0.0
                self.target_angular_vel = 0.0
            
            # 重置
            elif key.lower() == 'r':
                self.boom_pos = 0.0
                self.arm_pos = 0.0
                self.bucket_pos = 0.0
                self.swing_pos = 0.0
                print("\n[INFO] 所有关节已重置")
            
            # 调整步进量
            elif key == '+' or key == '=':
                self.step = min(self.step + 0.01, 0.5)
                print(f"\n步进量: {self.step:.3f}")
            elif key == '-':
                self.step = max(self.step - 0.01, 0.01)
                print(f"\n步进量: {self.step:.3f}")
            
            elif key.lower() == 'h':
                self.print_help()
            
            # 限制关节范围
            self.boom_pos = max(-1.0, min(0.5, self.boom_pos))
            self.arm_pos = max(-1.0, min(2.5, self.arm_pos))
            self.bucket_pos = max(-1.5, min(1.5, self.bucket_pos))
            self.swing_pos = max(-3.14, min(3.14, self.swing_pos))

            # 速度限制
            self.target_linear_vel = max(-self.max_linear_vel, min(self.max_linear_vel, self.target_linear_vel))
            self.target_angular_vel = max(-self.max_angular_vel, min(self.max_angular_vel, self.target_angular_vel))
            
            # 发布命令
            self.pub_boom.publish(Float64(self.boom_pos))
            self.pub_arm.publish(Float64(self.arm_pos))
            self.pub_bucket.publish(Float64(self.bucket_pos))
            self.pub_swing.publish(Float64(self.swing_pos))
            twist_cmd = Twist()
            twist_cmd.linear.x = self.target_linear_vel
            twist_cmd.linear.y = 0.0
            twist_cmd.linear.z = 0.0
            twist_cmd.angular.x = 0.0
            twist_cmd.angular.y = 0.0
            twist_cmd.angular.z = self.target_angular_vel
            self.pub_cmd_vel.publish(twist_cmd)
            
            # 显示当前位置
            sys.stdout.write(f"\r大臂:{self.boom_pos:+.2f} 小臂:{self.arm_pos:+.2f} 铲斗:{self.bucket_pos:+.2f} 回转:{self.swing_pos:+.2f} 底盘线速度:{self.target_linear_vel:+.2f} 角速度:{self.target_angular_vel:+.2f} ")
            sys.stdout.flush()

if __name__ == '__main__':
    try:
        controller = KeyboardController()
        controller.run()
    except rospy.ROSInterruptException:
        pass