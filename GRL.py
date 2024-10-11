import torch
from torch.autograd import Function

# 定义 Gradient Reversal Function
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        # 保存 lambda 值，lambda_ 决定反转的力度
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # 反转梯度，乘以 -lambda_
        lambda_ = ctx.lambda_
        grad_input = grad_output.neg() * lambda_
        return grad_input, None

# 定义 Gradient Reversal Layer
class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
