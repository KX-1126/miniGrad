from mlp import mlp

network = mlp(3, [4, 4, 2])

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
for p in ypred:
    print(p)    