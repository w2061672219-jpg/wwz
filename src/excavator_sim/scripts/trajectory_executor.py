#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import csv
import rospy
import numpy as np
from std_msgs.msg import Float64

# 假定你已经有这个类：从前面的回答里复制过来
from kinematics import ExcavatorKinematics


class FuxiWriter(object):
    def __init__(self):
        rospy.init_node("write_fuxi_trajectory")

        # === 1. 载入 / 写死轨迹配置 ===
        self.cfg = self.build_config_from_hardcode()

        # === 2. 运动学对象 ===
        self.kin = ExcavatorKinematics()

        # === 3. 控制器 Publisher ===
        self.pub_swing = rospy.Publisher(
            "/excavator/swing_position_controller/command",
            Float64,
            queue_size=10,
        )
        self.pub_boom = rospy.Publisher(
            "/excavator/boom_position_controller/command",
            Float64,
            queue_size=10,
        )
        self.pub_arm = rospy.Publisher(
            "/excavator/arm_position_controller/command",
            Float64,
            queue_size=10,
        )
        self.pub_bucket = rospy.Publisher(
            "/excavator/bucket_position_controller/command",
            Float64,
            queue_size=10,
        )

        # === 4. 记录数据用 ===
        self.record_file = rospy.get_param("~record_file", "fuxi_trajectory.csv")
        self.csv_f = open(self.record_file, "w")
        self.csv_writer = csv.writer(self.csv_f)
        self.csv_writer.writerow(
            [
                "t",
                "cmd_swing",
                "cmd_boom",
                "cmd_arm",
                "cmd_bucket",
                "fk_x",
                "fk_y",
                "fk_z",
            ]
        )

        # 目标末端俯仰角（相对水平，负数是“略向下挖”）
        self.target_pitch = rospy.get_param("~target_pitch", -0.6)

        # 控制频率（Hz）
        self.rate_hz = rospy.get_param("~rate", 50.0)
        self.rate = rospy.Rate(self.rate_hz)

    # ------------------------------------------------------------
    # 把你给的 YAML 配置写成一个 Python dict
    # （如果你已经存为 .yaml，用 yaml.safe_load 读出来即可）
    # ------------------------------------------------------------
    def build_config_from_hardcode(self):
        cfg = {
            "writing_plane": {
                "z_height": 0.8,
                "x_offset": 4.0,
                "y_offset": 0.0,
                "scale": 2.0,
            },
            "stroke_gap": {
                "lift_height": 0.3,
                "move_speed": 0.5,
            },
            "fu_character": {
                "strokes": [
                    {"name": "pie", "points": [[0.0, 0.8, 0.0], [-0.3, 0.4, 0.0], [-0.5, 0.0, 0.0]]},
                    {"name": "heng", "points": [[-0.2, 0.6, 0.0], [0.4, 0.6, 0.0]]},
                    {"name": "pie2", "points": [[0.1, 0.6, 0.0], [-0.1, 0.3, 0.0], [-0.3, 0.0, 0.0]]},
                    {"name": "na", "points": [[0.1, 0.5, 0.0], [0.3, 0.3, 0.0], [0.5, 0.0, 0.0]]},
                    {"name": "dian", "points": [[0.3, 0.7, 0.0], [0.35, 0.65, 0.0]]},
                ]
            },
            "xi_character": {
                "x_offset": 1.2,
                "strokes": [
                    {"name": "heng1", "points": [[-0.3, 0.9, 0.0], [0.3, 0.9, 0.0]]},
                    {"name": "pie1", "points": [[-0.1, 0.9, 0.0], [-0.3, 0.6, 0.0]]},
                    {"name": "dian1", "points": [[0.1, 0.85, 0.0], [0.2, 0.7, 0.0]]},
                    {"name": "heng2", "points": [[-0.25, 0.6, 0.0], [0.25, 0.6, 0.0]]},
                    {"name": "shu", "points": [[0.0, 0.6, 0.0], [0.0, 0.3, 0.0]]},
                    {"name": "bottom_left", "points": [[-0.3, 0.3, 0.0], [-0.4, 0.0, 0.0]]},
                    {"name": "bottom_mid", "points": [[-0.1, 0.3, 0.0], [0.0, 0.0, 0.0]]},
                    {"name": "bottom_right", "points": [[0.2, 0.3, 0.0], [0.4, 0.0, 0.0]]},
                ],
            },
        }
        return cfg

    # ------------------------------------------------------------
    # 字形坐标 -> 世界坐标
    # ------------------------------------------------------------
    def letter_to_world(self, x_local, y_local, which_char, char_cfg):
        wp = self.cfg["writing_plane"]
        scale = wp["scale"]

        # 基础平移
        x = wp["x_offset"] + scale * x_local
        y = wp["y_offset"] + scale * y_local

        # “羲”字整体再往右偏 x_offset
        if which_char == "xi":
            x += scale * char_cfg.get("x_offset", 0.0)

        z = wp["z_height"]  # 写字平面高度
        return x, y, z

    # ------------------------------------------------------------
    # 在两点之间插 N 步，让轨迹更平滑一些
    # ------------------------------------------------------------
    def interpolate_segment(self, p0, p1, num_steps):
        p0 = np.array(p0, dtype=float)
        p1 = np.array(p1, dtype=float)
        for i in range(num_steps):
            t = float(i) / max(1, (num_steps - 1))
            yield (1 - t) * p0 + t * p1

    # ------------------------------------------------------------
    # 生成整个“伏羲”末端轨迹（世界坐标）
    # 返回：list of dict，每个元素包含：
    #   {"x":..., "y":..., "z":..., "pen": True/False}
    # pen=True 表示在写字，False 表示提笔移动
    # ------------------------------------------------------------
    def build_world_trajectory(self):
        traj = []

        wp = self.cfg["writing_plane"]
        stroke_gap = self.cfg["stroke_gap"]
        lift_height = stroke_gap["lift_height"]

        # 先写“伏”，再写“羲”
        for which_char in ["fu", "xi"]:
            if which_char == "fu":
                char_cfg = self.cfg["fu_character"]
            else:
                char_cfg = self.cfg["xi_character"]

            strokes = char_cfg["strokes"]

            prev_end = None
            for si, stroke in enumerate(strokes):
                pts_local = stroke["points"]

                # 1) 当前笔画所有点映射到世界坐标
                pts_world = []
                for (xl, yl, zl) in pts_local:
                    xw, yw, zw = self.letter_to_world(
                        xl, yl, which_char, char_cfg
                    )
                    pts_world.append([xw, yw, zw])

                # 2) 如果不是第一笔：提笔 -> 移动到下一笔起点 -> 落笔
                start_pt = pts_world[0]
                if prev_end is not None:
                    # 提笔
                    up_start = [prev_end[0], prev_end[1], wp["z_height"] + lift_height]
                    up_end = [start_pt[0], start_pt[1], wp["z_height"] + lift_height]

                    # prev_end -> 上方
                    for p in self.interpolate_segment(prev_end, up_start, 10):
                        traj.append({"x": p[0], "y": p[1], "z": p[2], "pen": False})
                    # 上方移动
                    for p in self.interpolate_segment(up_start, up_end, 10):
                        traj.append({"x": p[0], "y": p[1], "z": p[2], "pen": False})
                    # 落笔
                    for p in self.interpolate_segment(up_end, start_pt, 10):
                        traj.append({"x": p[0], "y": p[1], "z": p[2], "pen": False})

                # 3) 写当前这一笔：点之间插点，pen=True
                for i in range(len(pts_world) - 1):
                    p0 = pts_world[i]
                    p1 = pts_world[i + 1]
                    for p in self.interpolate_segment(p0, p1, 20):
                        traj.append({"x": p[0], "y": p[1], "z": p[2], "pen": True})

                prev_end = pts_world[-1]

        return traj

    # ------------------------------------------------------------
    # 执行轨迹
    # ------------------------------------------------------------
    def run(self):
        rospy.loginfo("Building world trajectory (Fuxi)...")
        traj = self.build_world_trajectory()
        rospy.loginfo("Total points in trajectory: %d", len(traj))

        # 等待控制器连接
        rospy.sleep(1.0)

        t0 = rospy.Time.now().to_sec()
        for i, pt in enumerate(traj):
            if rospy.is_shutdown():
                break

            x = pt["x"]
            y = pt["y"]
            z = pt["z"]

            # 逆运动学，得到关节角
            joints = self.kin.inverse_kinematics([x, y, z], self.target_pitch)
            if joints is None:
                rospy.logwarn("IK failed for point #%d: (%.3f, %.3f, %.3f)", i, x, y, z)
                # 可以选择跳过 / 停止，这里先跳过
                continue

            q_swing, q_boom, q_arm, q_bucket = joints

            # 发布到控制器
            self.pub_swing.publish(Float64(q_swing))
            self.pub_boom.publish(Float64(q_boom))
            self.pub_arm.publish(Float64(q_arm))
            self.pub_bucket.publish(Float64(q_bucket))

            # 用 FK 算一下当前命令对应的末端位置，用来记录轨迹
            fk_pos = self.kin.forward_kinematics(joints)

            t_now = rospy.Time.now().to_sec() - t0
            self.csv_writer.writerow(
                [
                    f"{t_now:.4f}",
                    f"{q_swing:.6f}",
                    f"{q_boom:.6f}",
                    f"{q_arm:.6f}",
                    f"{q_bucket:.6f}",
                    f"{fk_pos[0]:.4f}",
                    f"{fk_pos[1]:.4f}",
                    f"{fk_pos[2]:.4f}",
                ]
            )

            self.rate.sleep()

        rospy.loginfo("Fuxi trajectory finished.")
        self.csv_f.close()


if __name__ == "__main__":
    writer = FuxiWriter()
    writer.run()