import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories_2d(ref_traj, actual_traj, save_path="fuxi_2d_plot.png", flip_y=True):
    """
    绘制 XY 平面的 2D 轨迹图
    :param ref_traj: 期望轨迹列表 [[x,y,z], ...]
    :param actual_traj:  实际轨迹列表（从ROS TF获取）[[x,y,z], ...]
    :param save_path: 图片保存路径
    :param flip_y: 是否对Y轴镜像（默认True以改善显示效果）
    """
    # 调试输出
    print(f"[Plot Debug] ref_traj points: {len(ref_traj)}")
    print(f"[Plot Debug] actual_traj points: {len(actual_traj)}")
    
    if len(ref_traj) > 0:
        print(f"[Plot Debug] ref_traj sample: {ref_traj[:3]}")
    if len(actual_traj) > 0:
        print(f"[Plot Debug] actual_traj sample: {actual_traj[:3]}")
    
    # 转换为 numpy 数组方便切片
    ref = np.array(ref_traj) if len(ref_traj) > 0 else np.array([])
    actual = np.array(actual_traj) if len(actual_traj) > 0 else np.array([])

    # 对Y轴进行镜像（如果需要）
    if flip_y and len(ref) > 0:
        ref = ref.copy()
        ref[:, 1] = -ref[:, 1]  # 反转Y坐标
    if flip_y and len(actual) > 0:
        actual = actual.copy()
        actual[:, 1] = -actual[:, 1]  # 反转Y坐标

    plt.figure(figsize=(12, 10))
    
    # === 绘制逻辑 ===
    # 假设 X 是挖掘机前方距离，Y 是挖掘机左右偏移
    # 绘图时通常习惯：横轴为 Y (左右)，纵轴为 X (前后) -> 这是一个俯视图
    
    # 1. 绘制期望轨迹 (红色实线，更粗更明显)
    if len(ref) > 0:
        plt.plot(ref[:, 1], ref[:, 0], 'r-', label='Reference (Expected)', linewidth=1.5, alpha=0.8, zorder=2)
        # 标记起点和终点
        plt.scatter(ref[0, 1], ref[0, 0], c='red', marker='o', s=100, zorder=4)
        plt.scatter(ref[-1, 1], ref[-1, 0], c='red', marker='s', s=100, zorder=4)
        plt.text(ref[0, 1]+0.1, ref[0, 0], " Start", fontsize=10, color='red', fontweight='bold')
    else:
        print("[Plot Warning] 期望轨迹为空!")

    # 2. 绘制实际轨迹 (蓝色实线，从ROS TF获取)
    if len(actual) > 0:
        plt.plot(actual[:, 1], actual[:, 0], 'b-', label='Actual (From ROS TF)', linewidth=1.5, alpha=0.6, zorder=2)
    else:
        print("[Plot Warning] 实际轨迹为空!")

    # === 格式美化 ===
    plt.title("Excavator Writing Trajectory - FuXi (伏羲) 2D Visualization", fontsize=14, fontweight='bold')
    plt.xlabel("Y Position (Left/Right) [m]", fontsize=12)
    plt.ylabel("X Position (Forward) [m]", fontsize=12)
    
    # 关键：强制坐标轴比例相等，否则汉字会变形
    plt.axis('equal') 
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='best')
    
    # 保存并关闭
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] 图片已保存至: {save_path}")
    # 如果你在本地运行且有图形界面，可以取消注释下面这行
    plt.show()
    plt.close()