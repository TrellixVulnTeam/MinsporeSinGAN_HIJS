import argparse
import warnings
from datetime import datetime
from glob import glob
from shutil import copyfile
from datasets.datasetgetter import get_dataset

import mindspore
import mindspore.nn

from models.generator import Generator
from models.discriminator import Discriminator

from train import *
from validation import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--data_dir', default='../data/', help='path to dataset')
parser.add_argument('--dataset', default='PHOTO',
                    help='type of dataset', choices=['PHOTO'])
parser.add_argument('--gantype', default='zerogp',
                    help='type of GAN loss', choices=['wgangp', 'zerogp', 'lsgan'])
parser.add_argument('--model_name', type=str,
                    default='SinGAN', help='model name')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
parser.add_argument('--val_batch', default=1, type=int)
parser.add_argument('--img_size_max', default=250,
                    type=int, help='Input image size')
parser.add_argument('--img_size_min', default=25,
                    type=int, help='Input image size')
parser.add_argument('--img_to_use', default=-999, type=int,
                    help='Index of the input image to use < 6287')
parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on validation set')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')
parser.add_argument('--port', default='8888', type=str)


def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn(
            'You have chosen a specific GPU: GPU {}'.format(args.gpu))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ngpus_per_node = mindspore.communication.get_group_size(
        group='hccl_world_group')

    if args.load_model is None:
        args.model_name = '{}_{}'.format(
            args.model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    makedirs(os.path.join(args.log_dir, 'codes'))
    makedirs(os.path.join(args.log_dir, 'codes', 'models'))
    makedirs(args.res_dir)

    if args.load_model is None:
        pyfiles = glob("./*.py")
        modelfiles = glob('./models/*.py')
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)
        for py in modelfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))

    formatted_print('Total Number of GPUs:', ngpus_per_node)
    formatted_print('Total Number of Workers:', args.workers)
    formatted_print('Batch Size:', args.batch_size)
    formatted_print('Max image Size:', args.img_size_max)
    formatted_print('Min image Size:', args.img_size_min)
    formatted_print('Log DIR:', args.log_dir)
    formatted_print('Result DIR:', args.res_dir)
    formatted_print('GAN TYPE:', args.gantype)

    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if len(args.gpu) == 1:
        args.gpu = 0
    else:
        args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

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

    if args.gpu is not None:
        mindspore.context.set_context(device_target="GPU")
        networks = [x for x in networks]
    else:
        exit(1)

    discriminator, generator, = networks

    ######################
    # Loss and Optimizer #
    ######################
    d_opt = mindspore.nn.Adam(
        discriminator.sub_discriminators[0].parameters(), 5e-4, 0.5, 0.999)
    g_opt = mindspore.nn.Adam(
        generator.sub_generators[0].parameters(), 5e-4, 0.5, 0.999)

    ##############
    # Load model #
    ##############
    args.stage = 0
    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = mindspore.load_checkpoint(load_file)
            for _ in range(int(checkpoint['stage'])):
                generator.progress()
                discriminator.progress()
            networks = [discriminator, generator]

            if args.gpu is not None:
                mindspore.context.set_context(device_target="GPU")
                networks = [x for x in networks]

            discriminator, generator, = networks

            args.stage = checkpoint['stage']
            args.img_to_use = checkpoint['img_to_use']
            discriminator.load_state_dict(checkpoint['D_state_dict'])
            generator.load_state_dict(checkpoint['G_state_dict'])
            d_opt.load_state_dict(checkpoint['d_optimizer'])
            g_opt.load_state_dict(checkpoint['g_optimizer'])
            print("=> loaded checkpoint '{}' (stage {})"
                  .format(load_file, checkpoint['stage']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))

    cudnn.benchmark = True

    ###########
    # Dataset #
    ###########
    train_dataset, _ = get_dataset(args.dataset, args)
    train_sampler = None

    train_loader = mindspore.DatasetHelper(train_dataset, batch_size=args.batch_size,
                                           shuffle=(train_sampler is None), num_workers=args.workers,
                                           pin_memory=True, sampler=train_sampler)

    ######################
    # Validate and Train #
    ######################
    z_fix_list = [mindspore.ops.Pad(mindspore.ops.StandardNormal(
        args.batch_size, 3, args.size_list[0], args.size_list[0]), [5, 5, 5, 5], value=0)]
    zero_list = [mindspore.ops.Pad(mindspore.ops.Zeros(args.batch_size, 3, args.size_list[zeros_idx], args.size_list[zeros_idx]),
                                   [5, 5, 5, 5], value=0) for zeros_idx in range(1, args.num_scale + 1)]
    z_fix_list = z_fix_list + zero_list

    if args.validation:
        validateSinGAN(train_loader, networks, args.stage,
                       args, {"z_rec": z_fix_list})
        return
    elif args.test:
        validateSinGAN(train_loader, networks, args.stage,
                       args, {"z_rec": z_fix_list})
        return

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
        record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
        record_txt.write('DATASET\t:\t{}\n'.format(args.dataset))
        record_txt.write('GANTYPE\t:\t{}\n'.format(args.gantype))
        record_txt.write('IMGTOUSE\t:\t{}\n'.format(args.img_to_use))
        record_txt.close()

    for stage in range(args.stage, args.num_scale + 1):
        if args.distributed:
            train_sampler.set_epoch(stage)

        trainSinGAN(train_loader, networks, {
                    "d_opt": d_opt, "g_opt": g_opt}, stage, args, {"z_rec": z_fix_list})
        validateSinGAN(train_loader, networks, stage,
                       args, {"z_rec": z_fix_list})

        if args.distributed:
            discriminator.module.progress()
            generator.module.progress()
        else:
            discriminator.progress()
            generator.progress()

        networks = [discriminator, generator]

        if args.gpu is not None:
            mindspore.context.set_context(device_target="GPU")
            networks = [x for x in networks]

        discriminator, generator, = networks

        # Update the networks at finest scale
        for net_idx in range(generator.current_scale):
            for param in generator.sub_generators[net_idx].parameters():
                param.requires_grad = False
            for param in discriminator.sub_discriminators[net_idx].parameters():
                param.requires_grad = False

        d_opt = mindspore.nn.Adam(discriminator.sub_discriminators[discriminator.current_scale].parameters(),
                                  5e-4, 0.5, 0.999)
        g_opt = mindspore.nn.Adam(generator.sub_generators[generator.current_scale].parameters(),
                                  5e-4, 0.5, 0.999)

        ##############
        # Save model #
        ##############
        if stage == 0:
            check_list = open(os.path.join(
                args.log_dir, "checkpoint.txt"), "a+")
        save_checkpoint({
            'stage': stage + 1,
            'D_state_dict': discriminator.state_dict(),
            'G_state_dict': generator.state_dict(),
            'd_optimizer': d_opt.state_dict(),
            'g_optimizer': g_opt.state_dict(),
            'img_to_use': args.img_to_use
        }, check_list, args.log_dir, stage + 1)
        if stage == args.num_scale:
            check_list.close()


if __name__ == '__main__':
    main()
