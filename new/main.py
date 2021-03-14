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
parser.add_argument('--img_size_max', default=250,
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
    args = parser.parse_args()

    if args.load_model is None:
        args.model_name = '{}_{}'.format(
            args.model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')
    makedirs('./logs/' + args.model_name)
    makedirs('./results/' + args.model_name)

    args.log_dir = os.path.join('./logs', args.model_name)
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    makedirs(os.path.join(args.log_dir, 'codes'))
    makedirs(os.path.join(args.log_dir, 'codes', 'models'))
    makedirs(args.res_dir)

    formatted_print('Max image Size:', args.img_size_max)
    formatted_print('Min image Size:', args.img_size_min)
    formatted_print('Log DIR:', args.log_dir)
    formatted_print('Result DIR:', args.res_dir)
    formatted_print('GAN TYPE:', args.gantype)

    mindspore.context.set_context(device_id=args.device)
    main_worker(args)


def main_worker(args):
    args.stage = 0
    ################
    # Define model #
    ################
    # 4/3 : scale factor in the paper
    scale_factor = 4/3
    tmp_scale = args.img_size_max / args.img_size_min
    args.num_scale = int(np.round(np.log(tmp_scale) / np.log(scale_factor)))

    args.size_list = [int(args.img_size_min * scale_factor**i)
                      for i in range(args.num_scale + 1)]

    discriminator = Discriminator()
    generator = Generator(args.img_size_min, args.num_scale, scale_factor)

    networks = [discriminator, generator]

    ######################
    # Loss and Optimizer #
    ######################
    d_opt = mindspore.nn.Adam(
        discriminator.sub_discriminators[0].get_parameters(), 5e-4, 0.5, 0.999)
    g_opt = mindspore.nn.Adam(
        generator.sub_generators[0].get_parameters(), 5e-4, 0.5, 0.999)

    #############
    # Set stage #
    #############
    args.stage = 0

    ###########
    # Dataset #
    ###########
    dataset = create_dataset()  # MSP: result is a Tensor

    ######################
    # Validate and Train #
    ######################
    args.batch_size = 1
    pad = mindspore.ops.Pad(((0, 0), (0, 0), (5, 5), (5, 5)))

    # 启动训练的Tensor，使用randn填充
    z_fix_list = [pad(
        Tensor(
            np.random.randn(args.batch_size, 3, args.size_list[0], args.size_list[0]), mindspore.float32
        )
    )]
    # 扩充后，负责存放每个梯度的数据的Tensor，使用zero填充
    zero_list = [pad(
        Tensor(
            np.zeros(
                (args.batch_size, 3, args.size_list[zeros_idx], args.size_list[zeros_idx])
            ), mindspore.float32
        )
    ) for zeros_idx in range(1, args.num_scale + 1)]

    z_fix_list = z_fix_list + zero_list

    networks = [discriminator, generator]

    check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
    record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
    record_txt.write('GANTYPE\t:\t{}\n'.format(args.gantype))
    record_txt.close()

    for stage in range(args.stage, args.num_scale + 1):
        trainSinGAN(dataset, networks, {
                    "d_opt": d_opt, "g_opt": g_opt}, stage, args, {"z_rec": z_fix_list})
        validateSinGAN(dataset, networks, stage,
                       args, {"z_rec": z_fix_list})
        discriminator.progress()
        generator.progress()

        # Update the networks at finest scale
        d_opt = mindspore.nn.Adam(discriminator.sub_discriminators[discriminator.current_scale].parameters(),
                                  5e-4, 0.5, 0.999)
        g_opt = mindspore.nn.Adam(generator.sub_generators[generator.current_scale].parameters(),
                                  5e-4, 0.5, 0.999)


if __name__ == '__main__':
    main()
