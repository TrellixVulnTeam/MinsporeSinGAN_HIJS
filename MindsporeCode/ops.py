import torch
from torch import autograd

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

# from inception import InceptionV3
'''
   MSP: These two grad function, we can't transform them to mindspore...
'''


def compute_grad_gp(d_out, x_in):
    pass


def compute_grad_gp_wgan(D, x_real, x_fake, gpu):
    pass


def compute_grad_gp(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def compute_grad_gp_wgan(D, x_real, x_fake, gpu):
    alpha = torch.rand(x_real.size(0), 1, 1, 1)

    x_interpolate = ((1 - alpha) * x_real + alpha * x_fake).detach()
    x_interpolate.requires_grad = True
    d_inter_logit = D(x_interpolate)
    grad = torch.autograd.grad(d_inter_logit, x_interpolate,
                               grad_outputs=torch.ones_like(d_inter_logit), create_graph=True)[0]

    norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)

    d_gp = ((norm - 1) ** 2).mean()
    return d_gp
