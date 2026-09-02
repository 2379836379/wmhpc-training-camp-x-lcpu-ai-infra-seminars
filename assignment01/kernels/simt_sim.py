"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""


def run(program):
    # 初始化32个lane的寄存器
    regs = list(range(32))
    cycles = 0
    # prog是元组
    def execute_prog(prog, mask):
        """执行一段程序，只对mask中为True的lane生效"""
        nonlocal cycles
        pc = 0
        while pc < len(prog):
            instr = prog[pc]
            op = instr[0]
            
            if op == "add":
                # 检查是否有active lane
                if any(mask):
                    k = instr[1]
                    for i in range(32):
                        if mask[i]:
                            regs[i] += k
                    cycles += 1
                pc += 1
                
            elif op == "mul":
                if any(mask):
                    k = instr[1]
                    for i in range(32):
                        if mask[i]:
                            regs[i] *= k
                    cycles += 1
                pc += 1
                
            elif op == "if_lt":
                # if_lt指令本身不计拍
                t = instr[1]
                then_prog = instr[2]
                else_prog = instr[3]
                
                # 计算then_mask和else_mask
                then_mask = [False] * 32
                else_mask = [False] * 32
                for i in range(32):
                    if mask[i]:  # 只考虑当前active的lane
                        if regs[i] < t:
                            then_mask[i] = True
                        else:
                            else_mask[i] = True
                
                # 执行then分支（如果有active lane）
                if any(then_mask):
                    execute_prog(then_prog, then_mask)
                
                # 执行else分支（如果有active lane）
                if any(else_mask):
                    execute_prog(else_prog, else_mask)
                
                # 分支汇合，继续执行下一条指令
                pc += 1
    
    # 初始所有lane都active
    initial_mask = [True] * 32
    execute_prog(program, initial_mask)
    
    return regs, cycles
