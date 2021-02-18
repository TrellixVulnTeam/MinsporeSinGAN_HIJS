<font face="Consolas">

# <font face="Cambria Math">cublog</font>

## 项目介绍

本项目旨在移植Pytorch上的SinGAN算法到Mindspore平台中

## 项目结构

### 使用的开发框架

- Mindspore

### 分包设计
- ./src:  原项目中将其设置为输出端，很令人迷惑，但是我们沿用这样的设计
- ./code: 承载所有的代码
- ./code/models: discriminator和generator
- ./code/main.py: 程序入口
- ./code/ops.py:  许多算子
- ./code/utils.py: 一些小工具，共享的代码
-  ./code/validation.py: 命令行参数带有validate时，需要调用的validateSinGAN函数
- ./TEMP: 修改到一半的代码

## 项目规范

使用 vscode 默认的排版格式进行格式化。

## 进展安排

本项目的想法：先分析整体的代码结构，之后着手修改代码。
