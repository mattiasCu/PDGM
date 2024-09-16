import torch
import torch.nn as nn
import torch.optim as optim

EPSILON = 1e-10

class SubsetOperator(nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON], device=scores.device))
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=2)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=2)
            khot_hard = khot_hard.scatter_(2, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

class SimpleModel(nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SimpleModel, self).__init__()
        self.subset_operator = SubsetOperator(k, tau, hard)

    def forward(self, x):
        x = self.subset_operator(x)
        return x

# 生成训练数据
def generate_data(batch_size, dim1, dim2, k):
    data = torch.rand(batch_size, dim1, dim2).cuda()*1000  # 生成随机输入数据
    labels = torch.zeros_like(data)
    _, indices = torch.topk(data, k, dim=2)
    labels.scatter_(2, indices, 1.0)  # 设置top-k位置为1
    return data, labels

# 设置参数
batch_size = 150
dim1 = 10
dim2 = 30
k = 3
tau = 1.0
hard = True
learning_rate = 0.001
num_epochs = 50

# 初始化模型、损失函数和优化器
model = SimpleModel( k, tau, hard).cuda()
criterion = nn.BCEWithLogitsLoss()  # 使用BCE损失函数来匹配形状

# 训练循环
for epoch in range(num_epochs):
    model.train()
    
    data, labels = generate_data(batch_size, dim1, dim2, k)
    outputs = model(data)
    loss = criterion(outputs, labels)
    #loss.backward()
    #optimizer.step()
    
    # 计算准确率
    with torch.no_grad():
        predicted = torch.sigmoid(outputs).round()
        accuracy = (predicted == labels).float().mean().item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

print("Training complete.")
