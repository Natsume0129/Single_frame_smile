import numpy as np

# 加载 npz 文件
# 'data.npz' 请替换为你的文件名
data = np.load('E:\Single_frame_smile\out_smilingornot\out_smilingornot\embeddings.npz')

# 1. 查看文件中包含的所有数组名称（Key）
# npz 文件类似于一个字典，存储了多个数组
print("包含的数组列表:", data.files)

# 2. 遍历并打印每个数组的具体信息
for file_name in data.files:
    array_data = data[file_name]
    print(f"\n数组名称: {file_name}")
    print(f"形状 (Shape): {array_data.shape}")
    print(f"数据类型 (Dtype): {array_data.dtype}")
    # 如果想看具体数值，可以打印 array_data
    # print(array_data)

# 3. 记得关闭文件（或者使用 with 语句）
data.close()