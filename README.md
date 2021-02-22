<font face="Consolas">

# <font face="Cambria Math">cublog</font>

## 项目介绍

本项目旨在移植 Pytorch 上的 SinGAN 算法到 Mindspore 平台中

## 项目结构

### 使用的开发框架

- Mindspore

### 分包设计

- ./src: 原项目中将其设置为输出端，很令人迷惑，但是我们沿用这样的设计
- ./code: 承载所有的代码
- ./code/models: discriminator 和 generator
- ./code/main.py: 程序入口
- ./code/ops.py: 许多算子
- ./code/utils.py: 一些小工具，共享的代码
- ./code/validation.py: 命令行参数带有 validate 时，需要调用的 validateSinGAN 函数
- ./TEMP: 修改到一半的代码

## 项目规范

使用 vscode 默认的排版格式进行格式化。

## 进展安排

本项目的想法：先分析整体的代码结构，之后着手修改代码。

## 如何启动

启动文件：./code/main.py
命令行参数：
| 参数           | 作用                                   | 默认值     |
| -------------- | -------------------------------------- | ---------- |
| --data_dir     | path to dataset                        | ../data/   |
| --data_set     | type of dataset                        | PHOTO      |
| --gantype      | type of GAN Loss                       | zerogp     |
| --model_name   | define the name of the model           | SinGAN     |
| --workers      | number of data loading workers         | 8          |
| --batch_size   | total batch size                       | 1          |
| --val_batch    |                                        | 1          |
| --img_size_max | input image size                       | 250        |
| --img_size_min | input image size                       | 25         |
| --img_to_use   | index of the input image to use < 6287 | -999       |
| --load_model   | path to latest checkpoint              | None       |
| --validation   | evaluate model on validation set       | validation |
| --test         | test model on validation set           | test       |
| --gpu          | gpu id to use                          | None       |
示范：
python main.py --gpu 0 --img_to_use 0 --img_size_max 1025 --gantype wgangp