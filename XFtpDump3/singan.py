import mindspore
import numpy as np
import torch
from ops import *
from mindspore import Tensor, nn, ops

######################################
# Comment:                           #
# lossCreator = generatorLossCreator #
# generator = generator              #
# opts = g_opt                       #
######################################
class TrainOneStepG(nn.Cell):
    def __init__(self, args, lossCreator, generator, discriminator, opts):
        super(TrainOneStepG, self).__init__(auto_prefix=False)
        self.args = args
        self.lossCreator = lossCreator
        self.weight = mindspore.ParameterTuple(
            generator.sub_generators[generator.current_scale].trainable_params())
        self.opts = opts
        self.generator = generator
        self.discriminator = discriminator
        self.lossCell = WithLossCell(lossCreator)

    def construct(self, x_in, z_rec, stage):
        # 通过生成器生成图像
        x_rec_list = self.generator(z_rec)
        # 计算Loss
        g_rec = nn.MSELoss()(x_rec_list[-1], x_in)
        # calculate rmse for each scale
        rmse_list = [1.0]
        for rmseidx in range(1, stage + 1):
            rmse = mindspore.ops.Sqrt(nn.MSELoss()(
                x_rec_list[rmseidx], self.x_in_list[rmseidx]))
            rmse_list.append(rmse)
        pad = mindspore.ops.Pad(((0, 0), (0, 0), (5, 5), (5, 5)))
        z_list = [pad(
            rmse_list[z_idx] *
            Tensor(np.random.randn(self.args.batch_size, 3,
                                   self.args.size_list[z_idx],
                                   self.args.size_list[z_idx]).astype(np.float32))
        )for z_idx in range(stage + 1)]
        x_fake_list = self.generator(z_list)
        g_fake_logit = self.discriminator(x_fake_list[-1])
        ones = mindspore.ops.OnesLike()(g_fake_logit)
        g_loss = self.lossCreator(x_in, g_fake_logit, g_rec, ones)
        # 计算反向梯度
        grad = ops.GradOperation(get_by_list=True)(self.lossCell, self.weight)
        grads_g = grad(x_in, g_fake_logit, g_rec, ones)
        return g_loss, rmse_list, z_list, ops.depend(g_loss, self.opts(grads_g))

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.

    Args:
        network (Cell): The target network to wrap.
    """

    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, x_in, g_fake_logit, g_rec, ones):
        lg = self.network(x_in, g_fake_logit, g_rec, ones)
        return lg  # loss

class GeneratorLoss(nn.Cell):
    def __init__(self, args):
        super(GeneratorLoss, self).__init__()
        self.args = args

    def construct(self, x_in, g_fake_logit, g_rec, ones):
        # 根据所选择GAN类型不同，而选择不一样的算法
        if self.args.gantype == 'wgangp':
            # wgan gp
            g_fake = -mindspore.ops.ReduceMean()(g_fake_logit, (2, 3))
            g_loss = g_fake + 10.0 * g_rec
        elif self.args.gantype == 'zerogp':
            # zero centered GP
            g_fake = mindspore.ops.BinaryCrossEntropy(
                reduction='none')(g_fake_logit, ones, None).mean()
            g_loss = g_fake + 100.0 * g_rec
        elif self.args.gantype == 'lsgan':
            # lsgan
            g_fake = nn.MSELoss()(mindspore.ops.ReduceMean(
                g_fake_logit, (2, 3)), 0.9 * ones)
            g_loss = g_fake + 50.0 * g_rec
        return g_loss

class TrainOneStepD(nn.Cell):
    def __init__(self, lossCreator, discriminator, opts):
        super(TrainOneStepD, self).__init__(auto_prefix=False)
        self.discriminator = discriminator
        self.lossCreator = lossCreator
        self.opts = opts

    def construct(self, x_in):
        d_loss = self.lossCreator(x_in)
        grad = ops.GradOperation()(self.discriminator)
        return d_loss, ops.depend(d_loss, self.opts(grad((x_in))))

class DiscriminatorLoss(nn.Cell):
    def __init__(self, G, D, args, z_list):
        super(DiscriminatorLoss, self).__init__()
        self.args = args
        self.D = D
        self.G = G
        self.G.set_grad()
        self.D.set_grad()
        self.z_list = z_list

    def construct(self, x_in):
        x_in.requires_grad = True
        x_fake_list = self.G(self.z_list)

        d_fake_logit = self.D(x_fake_list[-1])
        d_real_logit = self.D(x_in)

        ones = mindspore.ops.OnesLike()(d_real_logit)
        zeros = mindspore.ops.ZerosLike()(d_fake_logit)

        # 根据所选择GAN类型不同，而选择不一样的算法
        if self.args.gantype == 'wgangp':
            # wgan gp
            d_fake = mindspore.ops.ReduceMean(d_fake_logit, (2, 3))
            d_real = -mindspore.ops.ReduceMean(d_real_logit, (2, 3))
            d_gp = compute_grad_gp_wgan(
                self.D, x_in, x_fake_list[-1], self.args.gpu)
            d_loss = d_real + d_fake + 0.1 * d_gp
        elif self.args.gantype == 'zerogp':
            # zero centered GP
            d_fake = mindspore.ops.BinaryCrossEntropy(
                reduction='none')(d_fake_logit, zeros, None).mean()
            d_real = mindspore.ops.BinaryCrossEntropy(
                reduction='none')(d_real_logit, ones, None).mean()
            d_gp = compute_grad_gp(
                d_real_logit.mean((2, 3)), x_in)
            d_loss = d_real + d_fake + 10.0 * d_gp
        elif self.args.gantype == 'lsgan':
            # lsgan
            d_fake = nn.MSELoss()(mindspore.ops.ReduceMean(
                d_fake_logit, (2, 3)), zeros)
            d_real = nn.MSELoss()(mindspore.ops.ReduceMean(
                d_real_logit, (2, 3)), 0.9 * ones)
            d_loss = d_real + d_fake

        return d_loss, None, None
