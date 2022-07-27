import torch
import numpy as np
from torch.nn import functional as F

def cal_grad_penalty(critic, real_samples, fake_samples):
    """计算critic的惩罚项"""
    # 定义alpha
    alpha = torch.Tensor(np.random.randn(real_samples.size(0), 1, 1, 1)).cuda()

    # 从真实数据和生成数据中的连线采样
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).cuda()
    d_interpolates = critic(interpolates)  # 输出维度：[B, 1]


    fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = fake.cuda()

    # 对采样数据进行求导
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # 返回一个元组(value, )

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean().cuda()

    return gradient_penalty

def quantizer(w, L, device):
    top = (L-1)//2
    down = top - L + 1

    [B, W, H, C] = w.shape
    centers = torch.range(down, top).type(torch.FloatTensor).to(device)

    centers_stack = centers.reshape(1, 1, 1, 1, L)
    centers_stack = centers_stack.repeat(B, W, H, C, 1)
    # Partition W into the Voronoi tesellation over the centers
    w = w.reshape(B, W, H, C, 1)
    w_stack = w.repeat(1, 1, 1, 1, L)

    w_hard = torch.argmin(torch.abs(w_stack - centers_stack), dim=-1) + down #hard quantization

    smx = F.softmax(1.0 / torch.abs(w_stack - centers_stack + 10e-7), dim=-1)

    # Contract last dimension
    w_soft = torch.einsum('ijklm,m->ijkl', smx, centers)  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))

    # Treat quantization as differentiable for optimization
    w_detch = (w_hard - w_soft).clone().detach().requires_grad_(False)
    w_bar = torch.round(w_detch + w_soft)

    return w_bar

