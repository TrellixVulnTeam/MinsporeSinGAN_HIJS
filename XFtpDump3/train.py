import mindspore
from mindspore import ops
import mindspore.nn as nn
from utils import *
from tqdm import trange
import torchvision.utils as vutils
from ops import maxminnorm
from PIL import Image
from mindspore import Tensor
import numpy as np

from singan import *

def trainSinGAN(data_loader, networks, opts, stage, args, additional):
    # avg meter
    d_losses = AverageMeter()
    g_losses = AverageMeter()
    # set nets
    D = networks[0]
    G = networks[1]
    # set opts
    d_opt = opts['d_opt']
    g_opt = opts['g_opt']
    # switch to train mode
    D.set_train()
    G.set_train()
    # summary writer
    total_iter = 2000
    decay_lr = 1600

    d_iter = 3
    g_iter = 3
    # 显示进度条
    t_train = trange(0, total_iter, initial=0, total=total_iter)
    t_train.set_description('Stage: [{}/{}] Avg Loss: D[{d_losses.avg:.3f}] G[{g_losses.avg:.3f}] RMSE[{rmse}]'
                            .format(stage, args.num_scale, d_losses=d_losses, g_losses=g_losses, rmse='unknow'))

    z_rec = additional['z_rec']

    x_in = data_loader
    x_org = x_in
    x_in = mindspore.ops.ResizeBilinear(
        (args.size_list[stage], args.size_list[stage]), align_corners=True)(x_in)
    x_in = x_in.asnumpy()  # (1, 3, 25, 25)
    # 归一化(可能次序有问题)
    x_in = maxminnorm(x_in)
    # 减少维度
    x_in = x_in.squeeze()  # (3, 25, 25)
    # 保存numpy为图片
    temp_x = np.swapaxes(x_in, 0, 1)
    temp_x = np.swapaxes(temp_x, 1, 2)
    im = Image.fromarray(temp_x, 'RGB')
    im.save(os.path.join(args.res_dir, 'ORGTRAIN_{}.png'.format(stage)))
    # 恢复增加维度
    x_in = Tensor(x_in[np.newaxis, :], mindspore.float32)
    x_in_list = [x_in]
    for xidx in range(1, stage + 1):
        x_tmp = mindspore.ops.ResizeBilinear(
            (args.size_list[xidx], args.size_list[xidx]), align_corners=True)(x_org)
        x_in_list.append(x_tmp)

    # 训练一个梯度
    for i in t_train:
        # Generator train one step
        for _ in range(g_iter):
            ####### TODO: MAYBE FINISHED #######
            ####### 1. GeneratorLoss closure create    #######
            generatorLossCreator = GeneratorLoss(args)
            ####### 2. TrainOneStep  closure create    #######
            trainOneStepG = TrainOneStepG(args, generatorLossCreator, G, D, g_opt)
            ####### 3. TrainOneStep  invoked           #######
            g_loss, rmse_list, z_list, dependResultG = trainOneStepG(x_in, z_rec, stage)
            ####### 4. Update g_lossses to update tqdm #######
            g_losses.update(g_loss, x_in.shape[-1])

        # # Discriminator train one step
        # for _ in range(d_iter):
        #     ####### TODO: NOT FINISHED #######
        #     ####### 1. GeneratorLoss closure create    #######
        #     discriminatorLossCreator = DiscriminatorLoss(G, D, args, z_list)
        #     ####### 2. TrainOneStep  closure create    #######
        #     trainOneStepD = TrainOneStepD(discriminatorLossCreator, D, d_opt)
        #     ####### 3. TrainOneStep  invoked           #######
        #     d_loss, dependResultD = trainOneStepD(x_in)
        #     ####### 4. Update g_lossses to update tqdm #######
        #     d_losses.update(g_loss, x_in.shape[-1])

        # Finally, Update Description
        t_train.set_description(f'Stage: [{stage}/{args.num_scale}] Avg Loss: D[{d_losses.avg}] G[{g_losses.avg}] RMSE[{rmse_list[-1]}]')
