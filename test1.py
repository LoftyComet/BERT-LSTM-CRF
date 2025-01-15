import numpy as np
import matplotlib.pyplot as plt

# 假设我们有一些原始数据
data = np.array([1, 3, 2, 7, 9, 8, 6, 4, 5, 4])
window_size = 3

# 移动平均计算
smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 让我们分解这个过程：
# 1. np.ones(window_size) 创建一个全是1的数组：[1, 1, 1]
# 2. np.ones(window_size)/window_size 创建权重数组：[1/3, 1/3, 1/3]
# 3. np.convolve 进行卷积操作，对于每个位置：
#    - 取原数据中相邻的 window_size 个数
#    - 与权重数组相乘并求和
#    例如，第一个平滑值 = (1 + 3 + 2) / 3 = 2

# 可视化对比
plt.figure(figsize=(10, 5))
plt.plot(data, label='Original Data', marker='o')
plt.plot(range(window_size-1, len(data)), smoothed_data, 
         label='Smoothed Data', marker='o')
plt.legend()
plt.grid(True)
plt.show()

# 打印详细结果
print("原始数据:", data)
print("平滑后数据:", smoothed_data)

# 展示具体计算过程
print("\n计算过程示例：")
for i in range(len(smoothed_data)):
    window = data[i:i+window_size]
    result = np.mean(window)
    print(f"位置 {i}: {window} 的平均值 = {result}")