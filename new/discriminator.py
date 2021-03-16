from mindspore import nn
import mindspore
# super(FooChild,self) 首先找到 FooChild 的父类（就是类 nn.Module），然后把类 FooChild 的对象转换为类 nn.Module 的对象  nn.Module(向前传播)
true = True
class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.nf = 32
        self.current_scale = 0

        self.sub_discriminators = nn.CellList()

        first_discriminator = nn.CellList()#list

        first_discriminator.append(nn.SequentialCell(nn.Conv2d(3, self.nf, 3, 1, padding=1, has_bias=true, pad_mode='pad'),
                                                     nn.LeakyReLU(2e-1)))#SequentialCell 容器
        for _ in range(3):
            first_discriminator.append(nn.SequentialCell(nn.Conv2d(self.nf, self.nf, 3, 1, padding=1, has_bias=true, pad_mode='pad'),
                                                         nn.BatchNorm2d(self.nf),
                                                         nn.LeakyReLU(2e-1)))

        first_discriminator.append(nn.SequentialCell([nn.Conv2d(self.nf, 1, 3, 1, padding=1, has_bias=true, pad_mode='pad')]))

        first_discriminator = nn.SequentialCell(*first_discriminator)

        self.sub_discriminators.append(first_discriminator)
    
    def construct(self, x):
        out = self.sub_discriminators[self.current_scale](x)
        return out

    def progress(self):
        self.current_scale += 1
        # Lower scale discriminators are not used in later ... replace append to assign?
        if self.current_scale % 4 == 0:
            self.nf *= 2

        tmp_discriminator = nn.CellList()#tensor的list
        tmp_discriminator.append(nn.SequentialCell(nn.Conv2d(3, self.nf, 3, 1, padding=1, has_bias=true, pad_mode='pad'),
                                                  nn.LeakyReLU(2e-1)))

        for _ in range(3):
            tmp_discriminator.append(nn.SequentialCell(nn.Conv2d(self.nf, self.nf, 3, 1, padding=1, has_bias=true, pad_mode='pad'),
                                                   nn.BatchNorm2d(self.nf),
                                                   nn.LeakyReLU(2e-1)))

        tmp_discriminator.append(nn.SequentialCell(nn.Conv2d(self.nf, 1, 3, 1, padding=1, has_bias=true, pad_mode='pad')))

        tmp_discriminator = nn.SequentialCell(*tmp_discriminator)

        if self.current_scale % 4 != 0:
            prev_discriminator = self.sub_discriminators[-1]

            # Initialize layers via copy
            if self.current_scale >= 1:
                tmp_discriminator=mindspore.load_param_into_net(tmp_discriminator, prev_discriminator.parameters_dict)#以python的字典格式加载存储
                
        self.sub_discriminators.append(tmp_discriminator)
        print("DISCRIMINATOR PROGRESSION DONE")
