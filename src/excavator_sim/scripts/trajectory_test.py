#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import numpy as np
from std_msgs.msg import Float64

from kinematics import ExcavatorKinematics  # 用你前面那份类


def interpolate_line(p0, p1, n_steps):
    """在 p0 -> p1 之间线性插值 n_steps 个点（不含 p0，含 p1）"""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    points = []
    for i in range(1, n_steps + 1):
        t = float(i) / n_steps
        p = (1.0 - t) * p0 + t * p1
        points.append(p.tolist())
    return points


def main():
    rospy.init_node("move_bucket_tip_line")

    kin = ExcavatorKinematics()

    # === 1. 定义起点和终点（你可以在这里随便改） ===
    z_plane = 0.3
    p0 = [3.5, 0.0, z_plane]  # 起点
    p1 = [3.5, 2.0, z_plane]  # 终点

    # 目标铲斗俯仰角：例如保持水平（相对地面）
    target_pitch = None

    # === 2. 创建 Publisher （把 topic 改成你自己实际的） ===
    pub_swing = rospy.Publisher(
        "/excavator/swing_position_controller/command",
        Float64,
        queue_size=10,
    )
    pub_boom = rospy.Publisher(
        "/excavator/boom_position_controller/command",
        Float64,
        queue_size=10,
    )
    pub_arm = rospy.Publisher(
        "/excavator/arm_position_controller/command",
        Float64,
        queue_size=10,
    )
    pub_bucket = rospy.Publisher(
        "/excavator/bucket_position_controller/command",
        Float64,
        queue_size=10,
    )

    rate = rospy.Rate(50)  # 50Hz 控制

    # === 3. 先把铲斗尖端移到起点（单点 IK） ===
    q0 = kin.inverse_kinematics(p0, target_pitch=target_pitch)
    if q0 is None:
        rospy.logerr("起点不可达: %s", p0)
        return

    rospy.loginfo("Move to start point: %s, joints=%s", p0, np.round(q0, 3))

    for _ in range(200):  # 给一段时间让控制器走过去
        pub_swing.publish(Float64(q0[0]))
        pub_boom.publish(Float64(q0[1]))
        pub_arm.publish(Float64(q0[2]))
        pub_bucket.publish(Float64(q0[3]))
        rate.sleep()

    # === 4. 生成直线轨迹中的中间点 ===
    n_steps = 200  # 中间点数量，越大越平滑
    line_points = interpolate_line(p0, p1, n_steps)

    rospy.loginfo("Start moving along line: %s -> %s", p0, p1)

    # === 5. 逐点 IK -> 关节命令 ===
    for pt in line_points:
        if rospy.is_shutdown():
            break

        joints = kin.inverse_kinematics(pt, target_pitch=target_pitch)
        if joints is None:
            rospy.logwarn("目标点不可达: %s，跳过", pt)
            continue

        pub_swing.publish(Float64(joints[0]))
        pub_boom.publish(Float64(joints[1]))
        pub_arm.publish(Float64(joints[2]))
        pub_bucket.publish(Float64(joints[3]))

        rate.sleep()

    rospy.loginfo("Line motion finished.")


if __name__ == "__main__":
    main()