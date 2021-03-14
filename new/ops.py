import mindspore
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

def compute_grad_gp(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = mindspore.nn.WithGradCell(d_out)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def compute_grad_gp_wgan(D, x_real, x_fake, gpu):
    alpha = mindspore.ops.UniformReal(x_real.size(0), 1, 1, 1)

    x_interpolate = ((1 - alpha) * x_real + alpha * x_fake).detach()
    x_interpolate.requires_grad = True
    d_inter_logit = D(x_interpolate)
    grad = mindspore.nn.WithGradCell(d_inter_logit)[0]

    norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)

    d_gp = ((norm - 1) ** 2).mean()
    return d_gp
