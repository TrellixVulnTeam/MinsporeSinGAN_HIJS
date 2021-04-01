from mindspore import nn
from mindspore import Tensor
import mindspore

from generator import Generator
from discriminator import Discriminator

from utils import *
from train import *
from validation import *
from datasets import create_dataset


import argparse
import warnings
import numpy as np
from datetime import datetime
from glob import glob
from shutil import copyfile

parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--device', default=0,
                    type=int, help='Device we use')
parser.add_argument('--gantype', default='zerogp',
                    help='type of GAN loss', choices=['wgangp', 'zerogp', 'lsgan'])
parser.add_argument('--model_name', type=str,
                    default='SinGAN', help='model name')
parser.add_argument('--img_size_max', default=1025,
                    type=int, help='Input image size')
parser.add_argument('--img_size_min', default=25,
                    type=int, help='Input image size')
parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on validation set')


def main():
    # 初始处理 解决命令行参数的问题
    args = parser.parse_args()

    # 没指定LoadModel，就直接创造一个新的Model名
    if args.load_model is None:
        args.model_name = '{}_{}'.format(
            args.model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        args.model_name = args.load_model

    # 创建文件夹 存放Log和Results
    makedirs('./results/' + args.model_name)

    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.res_dir)

    # 回显 展示之前输入的命令行参数
    formatted_print('Max image Size:', args.img_size_max)
    formatted_print('Min image Size:', args.img_size_min)
    formatted_print('Result DIR:', args.res_dir)
    formatted_print('GAN TYPE:', args.gantype)

    # 设定使用哪个NPU
    mindspore.context.set_context(device_id=args.device)

    # 调用MainWorker，执行训练
    main_worker(args)


def main_worker(args):
    args.stage = 0
    ################
    # Define model #
    ################
    # 4/3 : scale factor in the paper
    # 4/3 : 论文中说的scale factor
    scale_factor = 4/3
    tmp_scale = args.img_size_max / args.img_size_min
    args.num_scale = int(np.round(np.log(tmp_scale) / np.log(scale_factor)))
    args.size_list = [int(args.img_size_min * scale_factor**i)
                      for i in range(args.num_scale + 1)]

    # 创建GAN算法中著名的discriminator和generator
    discriminator = Discriminator()
    generator = Generator(args.img_size_min, args.num_scale, scale_factor)

    # 将他们绑定到一起，设定为networks
    networks = [discriminator, generator]

    ######################
    # Loss and Optimizer #
    ######################
    # 设定D和G的optimizer
    d_opt = mindspore.nn.Adam(
        discriminator.sub_discriminators[0].trainable_params(), learning_rate=5e-4, beta1=0.5, beta2=0.999)
    g_opt = mindspore.nn.Adam(
        generator.sub_generators[0].trainable_params(), learning_rate=5e-4, beta1=0.5, beta2=0.999)
    #############
    # Set stage #
    #############
    args.stage = 0

    ###########
    # Dataset #
    ###########
    # 读取数据集，结果是：Tensor shape=(1, 3, max_size, max_size)
    dataset = create_dataset(args)
    ######################
    # Validate and Train #
    ######################
    # 设定Batch Size为1，一次只处理一张图片
    args.batch_size = 1
    pad = mindspore.nn.Pad(paddings=((0, 0), (0, 0), (5, 5), (5, 5)), mode="CONSTANT")

    # 启动训练的Tensor，使用randn填充
    z_fix_list = [pad(
        Tensor(
            np.random.randn(
                args.batch_size, 3, args.size_list[0], args.size_list[0]), mindspore.float32
        )
    )]
    # 扩充后，负责存放每个梯度的数据的Tensor，使用zero填充
    zero_list = [pad(
        Tensor(
            np.zeros(
                (args.batch_size, 3,
                 args.size_list[zeros_idx], args.size_list[zeros_idx])
            ), mindspore.float32
        )
    ) for zeros_idx in range(1, args.num_scale + 1)]

    z_fix_list = z_fix_list + zero_list

    # 循环训练 stage
    for stage in range(args.stage, args.num_scale + 1):
        trainSinGAN(dataset, networks, {
                        "d_opt": d_opt, "g_opt": g_opt
                    }, stage, args, {"z_rec": z_fix_list})
        # validateSinGAN(dataset, networks, stage,
        #                args, {"z_rec": z_fix_list})
        discriminator.progress()
        generator.progress()

        # Update the networks at finest scale
        # 更新opts
        d_opt = mindspore.nn.Adam(discriminator.sub_discriminators[discriminator.current_scale].trainable_params(),
                                  5e-4, 0.5, 0.999)
        g_opt = mindspore.nn.Adam(generator.sub_generators[generator.current_scale].trainable_params(),
                                  5e-4, 0.5, 0.999)


if __name__ == '__main__':
    main()
