from matplotlib import pyplot as plt


import pandas as pd

sum(range(1,11))

# 读取CSV文件
df = pd.read_csv('Eye.csv')

# 方法1：groupby后获取某列数据
# 假设按'group_column'分组，获取'target_column'的数据
grouped_data = df.groupby('Case')

# 按某列分组并获取特定列的值存入列表
eye_x = {}
eye_y = {}
eye_z = {}
head_x = {}
head_y = {}
head_z = {}
for name, group in df.groupby('Case'):
    eye_x[name] = group['GazeDirectionX'].tolist()
    eye_y[name] = group['GazeDirectionY'].tolist()
    eye_z[name] = group['GazeDirectionZ'].tolist()
    head_x[name] = group['HeadX'].tolist()
    head_y[name] = group['HeadY'].tolist()
    head_z[name] = group['HeadZ'].tolist()

temp = 3

eye_x1 = eye_x[temp]
eye_y1 = eye_y[temp]
eye_z1 = eye_z[temp]
head_x1 = head_x[temp]
head_y1 = head_y[temp]
head_z1 = head_z[temp]

import numpy as np


def calculate_kinematics(x_positions, y_positions, z_positions, time_step=1.0):
    """
    根据独立的x,y,z位置列表计算速度和加速度

    参数:
    x_positions: x轴位置列表
    y_positions: y轴位置列表
    z_positions: z轴位置列表
    time_step: 时间步长，默认为1.0

    返回:
    velocities_x, velocities_y, velocities_z: 三个方向的速度
    accelerations_x, accelerations_y, accelerations_z: 三个方向的加速度
    """

    def calculate_velocity(positions):
        velocities = np.zeros(len(positions))
        # 中间点使用中心差分
        velocities[1:-1] = (np.array(positions[2:]) - np.array(positions[:-2])) / (2 * time_step)
        # 边界点使用前向/后向差分
        velocities[0] = (positions[1] - positions[0]) / time_step
        velocities[-1] = (positions[-1] - positions[-2]) / time_step
        return velocities

    def calculate_acceleration(positions):
        accelerations = np.zeros(len(positions))
        # 中间点
        accelerations[1:-1] = (np.array(positions[2:]) - 2 * np.array(positions[1:-1])
                               + np.array(positions[:-2])) / (time_step ** 2)
        # 边界点
        accelerations[0] = (positions[2] - 2 * positions[1] + positions[0]) / (time_step ** 2)
        accelerations[-1] = (positions[-1] - 2 * positions[-2] + positions[-3]) / (time_step ** 2)
        return accelerations

    # 计算每个方向的速度
    velocities_x = calculate_velocity(x_positions)
    velocities_y = calculate_velocity(y_positions)
    velocities_z = calculate_velocity(z_positions)

    # 计算每个方向的加速度
    accelerations_x = calculate_acceleration(x_positions)
    accelerations_y = calculate_acceleration(y_positions)
    accelerations_z = calculate_acceleration(z_positions)

    return (velocities_x, velocities_y, velocities_z,
            accelerations_x, accelerations_y, accelerations_z)


def visualize_kinematics(x, y, z, time_step=1.0):
    """
    可视化运动轨迹、速度和加速度
    """
    # 计算时间点
    t = np.arange(len(x)) * time_step

    # 计算运动学量
    vx, vy, vz, ax, ay, az = calculate_kinematics(x, y, z, time_step)

    # 创建图形
    fig = plt.figure(figsize=(15, 10))

    # 1. 3D轨迹图
    # ax1 = fig.add_subplot(231, projection='3d')
    # ax1.plot(x, y, z, 'b-o', label='轨迹')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.set_zlabel('Z')
    # ax1.set_title('3Dtrajectory')

    # 2. 位置-时间图
    ax2 = fig.add_subplot(231)
    ax2.plot(t, x, 'r-', label='X')
    ax2.plot(t, y, 'g-', label='Y')
    ax2.plot(t, z, 'b-', label='Z')
    ax2.set_xlabel('time')
    ax2.set_ylabel('location')
    ax2.set_title('location-time')
    ax2.legend()

    # 3. 速度-时间图
    ax3 = fig.add_subplot(232)
    ax3.plot(t, vx, 'r-', label='Vx')
    ax3.plot(t, vy, 'g-', label='Vy')
    ax3.plot(t, vz, 'b-', label='Vz')
    ax3.set_xlabel('time')
    ax3.set_ylabel('Velocity')
    ax3.set_title('V-time')
    ax3.legend()

    # 4. 加速度-时间图
    ax4 = fig.add_subplot(233)
    ax4.plot(t, ax, 'r-', label='Ax')
    ax4.plot(t, ay, 'g-', label='Ay')
    ax4.plot(t, az, 'b-', label='Az')
    ax4.set_xlabel('time')
    ax4.set_ylabel('Acceleration')
    ax4.set_title('A-time')
    ax4.legend()

    # # 5. 速度矢量图（在XY平面上的投影）
    # ax5 = fig.add_subplot(235)
    # ax5.quiver(x[:-1], y[:-1], vx[:-1], vy[:-1],
    #            angles='xy', scale_units='xy', scale=1, color='r', label='速度矢量')
    # ax5.plot(x, y, 'b--', label='轨迹')
    # ax5.set_xlabel('X')
    # ax5.set_ylabel('Y')
    # ax5.set_title('速度矢量（XY平面投影）')
    # ax5.legend()
    #
    # # 6. 加速度矢量图（在XY平面上的投影）
    # ax6 = fig.add_subplot(236)
    # ax6.quiver(x[:-1], y[:-1], ax[:-1], ay[:-1],
    #            angles='xy', scale_units='xy', scale=1, color='g', label='加速度矢量')
    # ax6.plot(x, y, 'b--', label='轨迹')
    # ax6.set_xlabel('X')
    # ax6.set_ylabel('Y')
    # ax6.set_title('加速度矢量（XY平面投影）')
    # ax6.legend()

    plt.tight_layout()
    plt.show()


import numpy as np

import matplotlib.pyplot as plt


def plot_magnitude(t, speeds, acceleration_mags):
    """
    绘制速率和加速度大小随时间的变化
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 速率图
    ax1.plot(t, speeds, '-')
    ax1.set_xlabel('t(s)')
    ax1.set_ylabel('v (m/s)')
    ax1.set_title('v-t')
    ax1.grid(True)

    # 加速度大小图
    ax2.plot(t, acceleration_mags, '-')
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel('a (m/s²)')
    ax2.set_title('a-t')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()




def calculate_kinematics_magnitude(x_positions, y_positions, z_positions, time_step=1.0):
    """
    计算速度和加速度的标量大小

    返回:
    velocity_magnitudes: 速率（速度大小）
    acceleration_magnitudes: 加速度大小
    """

    def calculate_velocity(positions):
        velocities = np.zeros(len(positions))
        velocities[1:-1] = (np.array(positions[2:]) - np.array(positions[:-2])) / (2 * time_step)
        velocities[0] = (positions[1] - positions[0]) / time_step
        velocities[-1] = (positions[-1] - positions[-2]) / time_step
        return velocities

    def calculate_acceleration(positions):
        accelerations = np.zeros(len(positions))
        accelerations[1:-1] = (np.array(positions[2:]) - 2 * np.array(positions[1:-1])
                               + np.array(positions[:-2])) / (time_step ** 2)
        accelerations[0] = (positions[2] - 2 * positions[1] + positions[0]) / (time_step ** 2)
        accelerations[-1] = (positions[-1] - 2 * positions[-2] + positions[-3]) / (time_step ** 2)
        return accelerations

    # 计算各方向的速度和加速度
    vx = calculate_velocity(x_positions)
    vy = calculate_velocity(y_positions)
    vz = calculate_velocity(z_positions)

    ax = calculate_acceleration(x_positions)
    ay = calculate_acceleration(y_positions)
    az = calculate_acceleration(z_positions)

    # 计算速度大小（速率）
    velocity_magnitudes = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    # 计算加速度大小
    acceleration_magnitudes = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    return velocity_magnitudes, acceleration_magnitudes



speeds, acceleration_mags = calculate_kinematics_magnitude(eye_x1, eye_y1, eye_z1)
t = np.arange(len(eye_x1))  # 时间点

# 绘制图表
plot_magnitude(t, speeds, acceleration_mags)

speeds, acceleration_mags = calculate_kinematics_magnitude(head_x1, head_y1, head_z1)
t = np.arange(len(head_x1))  # 时间点

# 绘制图表
plot_magnitude(t, speeds, acceleration_mags)

# 示例使用
if __name__ == "__main__":
    # 测试数据
    x = [0, 1, 4, 9, 16]  # x轴位置
    y = [0, 2, 4, 6, 8]  # y轴位置
    z = [0, 1, 4, 9, 16]  # z轴位置

    # 计算速率和加速度大小
    speeds, acceleration_mags = calculate_kinematics_magnitude(x, y, z)

    # 打印结果
    print("时间点    速率(m/s)    加速度大小(m/s²)")
    print("-" * 40)
    for t in range(len(x)):
        print(f"t={t}    {speeds[t]:.2f}        {acceleration_mags[t]:.2f}")

    # 计算平均速率和平均加速度大小
    avg_speed = np.mean(speeds)
    avg_acceleration = np.mean(acceleration_mags)

    print("\n统计信息:")
    print(f"平均速率: {avg_speed:.2f} m/s")
    print(f"平均加速度大小: {avg_acceleration:.2f} m/s²")
    print(f"最大速率: {np.max(speeds):.2f} m/s")
    print(f"最大加速度大小: {np.max(acceleration_mags):.2f} m/s²")

# 计算速度和加速度
vx, vy, vz, ax, ay, az = calculate_kinematics(eye_x1, eye_y1, eye_z1)

visualize_kinematics(eye_x1, eye_y1, eye_z1)

visualize_kinematics(head_x1, head_y1, head_z1)

# 打印结果
print("速度:")
print(f"X方向: {vx}")
print(f"Y方向: {vy}")
print(f"Z方向: {vz}")

print("\n加速度:")
print(f"X方向: {ax}")
print(f"Y方向: {ay}")
print(f"Z方向: {az}")





fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))  # 创建 3 行 1 列的子图

axes[0].plot(eye_x1, label='eye_x')
axes[0].plot(head_x1, label='head_x')
axes[0].set_title('X coordinate')
# axes[0].set_xlabel('the number of samples')
axes[0].set_ylabel('X')
axes[0].legend()
# axes[0].grid(True)

axes[1].plot(eye_y1, label='eye_y')
axes[1].plot(head_y1, label='head_y')
axes[1].set_title('Y coordinate')
# axes[1].set_xlabel('the number of samples')
axes[1].set_ylabel('Y')
axes[1].legend()
# axes[1].grid(True)

axes[2].plot(eye_z1, label='eye_z')
axes[2].plot(head_z1, label='head_z')
axes[2].set_title('Z coordinate')
axes[2].set_xlabel('the number of samples')
axes[2].set_ylabel('Z')
axes[2].legend()
# axes[2].grid(True)
# 调整子图布局，避免重叠
plt.tight_layout()
# 显示图形
plt.show()