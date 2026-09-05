"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""
import torch
import tilelang
import tilelang.language as T


def make_softmax(M, N, dtype="float32"):
    """创建 Softmax kernel"""
    frag_width = 1
    while frag_width < N:
        frag_width <<= 1
    
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, 1), 1, threads=128) as (bx, by):
            row = bx
            X_frag = T.alloc_fragment((frag_width,), dtype)
            Y_frag = T.alloc_fragment((frag_width,), dtype)

            for i in T.Parallel(frag_width):
                X_frag[i] = T.if_then_else(i < N, X[row, i], -T.infinity(dtype))
            
            max_val = T.alloc_fragment((1,), dtype)
            T.reduce_max(X_frag, max_val, dim=0)
            
            for i in T.Parallel(frag_width):
                X_frag[i] = T.exp(X_frag[i] - max_val[0])
            
            sum_val = T.alloc_fragment((1,), dtype)
            T.reduce_sum(X_frag, sum_val, dim=0)
            
            for i in T.Parallel(frag_width):
                Y_frag[i] = T.if_then_else(i < N, X_frag[i] / sum_val[0], 0.0)
            
            for i in T.Parallel(frag_width):
                if i < N:
                    Y[row, i] = Y_frag[i]
    
    return main

_KERNEL_CACHE = {}
def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    对输入张量 x (M, N) 逐行做 softmax。
    """   
    M, N = x.shape
    key = (M, N, str(x.dtype), x.device.index)
    if key not in _KERNEL_CACHE:
      prim_func = make_softmax(M, N, dtype="float32")
      _KERNEL_CACHE[key] = tilelang.compile(prim_func, out_idx=[1])
      
    kernel = _KERNEL_CACHE[key]
    return kernel(x)