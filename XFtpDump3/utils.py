import mindspore
import os

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def formatted_print(notice, value):
    print('{0:<40}{1:<40}'.format(notice, value))

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
