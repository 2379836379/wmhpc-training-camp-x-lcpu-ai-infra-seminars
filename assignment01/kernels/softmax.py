"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    X_ptr,           # 输入指针
    Y_ptr,           # 输出指针
    M,               # 行数
    N,               # 列数
    stride_x_row,    # X 的行步长
    stride_y_row,    # Y 的行步长
    BLOCK_SIZE: tl.constexpr,  # block 大小（2 的幂）
):
    """
    Softmax kernel: 每个 program 处理一行
    """
    # 获取当前 program 处理的行索引
    row = tl.program_id(0)
    
    # 计算当前行的起始指针
    x_row_ptr = X_ptr + row * stride_x_row
    y_row_ptr = Y_ptr + row * stride_y_row
    
    # 创建列索引（从 0 到 BLOCK_SIZE-1）
    cols = tl.arange(0, BLOCK_SIZE)
    
    # 加载数据（mask 处理边界）
    mask = cols < N
    x = tl.load(x_row_ptr + cols, mask=mask, other=-float('inf'))
    
    # 计算最大值（数值稳定）
    max_val = tl.max(x, axis=0)
    
    # 减最大值并计算 exp
    x_exp = tl.exp(x - max_val)
    
    # 计算求和
    sum_val = tl.sum(x_exp, axis=0)
    
    # 归一化
    y = x_exp / sum_val
    
    # 写回结果（只写有效位置）
    tl.store(y_row_ptr + cols, y, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 计算 BLOCK_SIZE（不小于 N 的 2 的幂）
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # 计算网格大小：每行一个 program
    grid = (M,)
    
    # 启动 kernel
    softmax_kernel[grid](
        x, y, M, N,
        x.stride(0), y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y
