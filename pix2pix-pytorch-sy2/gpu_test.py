import torch
import time

# 确保 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建大矩阵
matrix_size = 10000
a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)

# 测试矩阵乘法速度
start_time = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()  # 确保所有计算完成
print(f"Time taken for matrix multiplication: {time.time() - start_time:.4f} seconds")