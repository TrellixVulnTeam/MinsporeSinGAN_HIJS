import os
import mindspore


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def formatted_print(notice, value):
    print('{0:<40}{1:<40}'.format(notice, value))


def save_checkpoint(state, check_list, log_dir, epoch=0):
    check_file = os.path.join(log_dir, 'model_{}.ckpt'.format(epoch))
    mindspore.save_checkpoint(state, check_file) # MSP: 潜在问题
    check_list.write('model_{}.ckpt\n'.format(epoch))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
