#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
from sensor_msgs.msg import JointState
# geometry_msgs 用于处理 TF 返回的向量数据
from geometry_msgs.msg import TransformStamped 

class ExcavatorMonitor:
    def __init__(self):
        # 1. 初始化节点
        rospy.init_node('excavator_status_monitor', anonymous=True)
        
        # 2. 关节角度存储变量
        self.joint_positions = {}
        
        # 3. 订阅关节状态话题
        self.sub_joints = rospy.Subscriber('/excavator/joint_states', JointState, self.joint_callback)
        
        # ==================== 核心：TF 设置 ====================
        # 4. 创建 TF 缓冲区和监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 5. 设置循环频率 (5Hz，即每秒刷新5次数据)
        self.rate = rospy.Rate(5) 

    def joint_callback(self, msg):
        """
        回调函数：当接收到关节状态时触发
        将关节名称和对应的角度值存入字典
        """
        try:
            for i, name in enumerate(msg.name):
                self.joint_positions[name] = round(msg.position[i], 3)
        except Exception as e:
            pass

    def get_end_effector_pose(self):
        """
        查询 TF 树，获取铲斗末端的坐标
        """
        try:
            # lookup_transform(target_frame, source_frame, time)
            # 含义：我想查询 "bucket" (目标) 相对于 "base_footprint" (源/基准) 的坐标
            # rospy.Time(0) 表示获取当前缓冲区里最新的那个变换
            trans = self.tf_buffer.lookup_transform('base_footprint', 'bucket_tip', rospy.Time(0))
            
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
            return x, y, z
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # 刚启动时可能 TF 树还没建立好，查询会失败，这里捕获异常防止程序崩溃
            return None, None, None

    def run(self):
        print("正在等待关节数据和 TF 变换...")
        while not rospy.is_shutdown():
            # 1. 获取末端坐标
            x, y, z = self.get_end_effector_pose()
            
            # 2. 清屏并打印信息 
            print("\033[H\033[J") 
            print("========================================")
            print("         挖掘机状态监控面板")
            print("========================================")
            
            print("\n【关节角度 (弧度)】:")
            if not self.joint_positions:
                print("  等待数据...")
            else:
                for name, pos in self.joint_positions.items():
                    # 格式化打印，保留2位小数
                    print(f"  {name:<25}: {pos:.2f} rad")

            print("\n【铲斗末端坐标 (相对于底盘)】:")
            if x is not None:
                print(f"  X (前后): {x:.2f} m")
                print(f"  Y (左右): {y:.2f} m")
                print(f"  Z (高度): {z:.2f} m")
            else:
                print("  TF 数据暂不可用 (请检查 robot_state_publisher 是否运行)")
                
            print("\n----------------------------------------")
            print("按 Ctrl+C 退出")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        monitor = ExcavatorMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass