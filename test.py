from mlp import mlp
from engine import Tensor

network = mlp(3, [4, 4, 1])

# 输入数据 (每个样本含 3 个特征)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.01],
    [1.2, 1.0, -1.0],
]

# 目标输出
ys = [1.0, -1.0, -1.0, 1.0]

# 前向预测
ypred = [network.forward(x) for x in xs]

# training
# 训练的本质：不断调整 w&b，使 gap 变小

step = 0.1

for i in range(20):
    ypred = [network.forward(x) for x in xs]
    gap = Tensor(0)

    for y, p in zip(ys, ypred):
        gap += (y - p)**2

    network.zero_grad()
    gap.backward()
    for p in network.parameters():
        # print(f"param: {p.data}, grad: {p.grad}")
        p.data += -p.grad * step 
    print(f"step {i}, gap: {gap}")
