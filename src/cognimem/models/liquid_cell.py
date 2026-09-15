import numpy as np

class LiquidCell:
    """
    闭式连续时间网络 (CfC) 单元 - 模拟液态时间常数 (LTC) 的行为。
    特点：具备时间依赖性的自适应动态，但使用封闭形式计算，避免了解微分方程的迭代开销。
    """

    def __init__(self, hidden_dim: int, time_constants: list = None):
        self.hidden_dim = hidden_dim
        # 液态时间常数 (模拟突触的适应性)
        if time_constants is None:
            self.tau = np.array([0.5, 1.0, 2.0])  # 默认多时间尺度
        else:
            self.tau = np.array(time_constants)

        # 初始化权重 (演示展示，使用高斯分布，后续可替换为训练好的权重)
        # 输入权重 (Input -> Hidden)
        self.W_input = np.random.randn(hidden_dim, hidden_dim) * 0.1
        # 循环权重 (Hidden -> Hidden)
        self.W_hidden = np.random.randn(hidden_dim, hidden_dim) * 0.1
        # 偏置
        self.bias = np.zeros(hidden_dim)

    def forward(self, input_vec: np.ndarray, state: np.ndarray = None) -> tuple:
        """
        前向传播：实现 CfC 的核心公式。
        公式模拟：h(t) = (1 - σ(...)) * h_prev + σ(...) * tanh(W_in * x + W_h * h_prev + b)
        其中 σ 是 sigmoid 门控，且受到时间常数 tau 的影响。
        """
        if state is None:
            state = np.zeros(self.hidden_dim)

        # 1. 计算基础的线性变换
        input_part = np.dot(self.W_input, input_vec)
        hidden_part = np.dot(self.W_hidden, state)
        combined = input_part + hidden_part + self.bias

        # 2. 液态门控机制 (受时间常数调节)
        # 灵感来源于 LTC：门控信号决定当前输入与历史状态的融合比例
        # 使用模拟的 tau 来调节 sigmoid 的陡峭程度，体现"时间感知"
        tau_factor = np.mean(self.tau)  # 简化：取平均时间常数作为全局调节

        # 输入门 (决定新信息的流入)
        input_gate = 1 / (1 + np.exp(-(combined / tau_factor)))
        # 遗忘门 (决定旧信息的保留)
        forget_gate = 1 - input_gate  # 互补门控，简化设计

        # 3. 候选隐藏状态 (非线性变换)
        candidate_state = np.tanh(combined)

        # 4. 新状态 = 遗忘门 * 旧状态 + 输入门 * 候选状态 (液态融合)
        new_state = forget_gate * state + input_gate * candidate_state

        # 输出即为新状态 (模仿生物学，所有神经元状态都可以视为输出)
        output = new_state

        return output, new_state

    def reset(self):
        """重置内部状态 (由外部Engine调用)"""
        pass