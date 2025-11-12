# PyTorch - PyTorch 深度学习框架

PyTorch 深度学习框架的学习资料库，涵盖张量基础、自动微分、神经网络实现、数据加载、训练循环、GPU 训练等核心主题。

## 📚 项目简介

本目录专注于 PyTorch 深度学习框架的学习与实践，深入解析 PyTorch 的核心组件、使用方法和最佳实践。内容涵盖从张量基础到多 GPU 训练的完整学习路径，适合 AI 工程师、深度学习研究者、算法开发者等技术人员学习和参考。

## 🗂️ 目录结构

```
PyTorch/
├── README.md                # 本文件，PyTorch 学习总览
├── pytorch-1h.ipynb        # PyTorch 一小时教程：从张量到在多GPU上训练神经网络
└── img/                    # 相关图片资源
```

## 📖 核心内容

### 1. PyTorch 一小时教程

**[PyTorch 一小时教程：从张量到在多GPU上训练神经网络](pytorch-1h.ipynb)** - 快速掌握 PyTorch 核心概念

**原文链接：** [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/)

**教程目标：**
- 用约一小时的阅读时间掌握 PyTorch 核心概念
- 快速上手深度神经网络实现
- 特别关注大语言模型（LLMs）等实际应用场景

**主要内容：**

#### 1.1 PyTorch 概述
- **PyTorch 的三个核心组件**
  - 张量库（Tensor Library）- 扩展 NumPy，支持 GPU 加速
  - 自动微分引擎（Autograd）- 自动计算梯度，简化反向传播
  - 深度学习库（Deep Learning Library）- 模块化构建块和预训练模型
- **深度学习定义** - AI、机器学习、深度学习的关系
- **安装 PyTorch** - CPU 和 GPU 版本的安装指南

#### 1.2 理解张量
- **标量、向量、矩阵和张量** - 多维数组数据结构
- **张量数据类型** - 不同精度和类型的选择
- **常见的 PyTorch 张量操作**
  - 创建张量（zeros, ones, randn, arange 等）
  - 张量运算（加法、乘法、矩阵运算等）
  - 形状操作（reshape, view, transpose 等）
  - 索引和切片
  - GPU 加速计算

#### 1.3 计算图
- **将模型视为计算图** - 理解前向传播和反向传播
- **动态计算图** - PyTorch 的动态特性优势

#### 1.4 自动微分
- **轻松实现自动微分** - Autograd 机制详解
- **反向传播** - 梯度计算和链式法则
- **梯度管理** - requires_grad、backward、grad 等

#### 1.5 神经网络实现
- **实现多层神经网络**
  - nn.Module 基类
  - 线性层、激活函数、损失函数
  - 优化器（SGD、Adam 等）
- **网络架构设计** - 构建自定义神经网络

#### 1.6 数据加载
- **设置高效的数据加载器**
  - Dataset 和 DataLoader
  - 数据预处理和转换
  - 批处理和采样策略
  - 多进程数据加载

#### 1.7 训练循环
- **典型的训练循环**
  - 前向传播
  - 损失计算
  - 反向传播
  - 参数更新
  - 验证和评估
- **训练技巧** - 学习率调度、早停、模型检查点

#### 1.8 模型保存和加载
- **保存和加载模型**
  - 保存完整模型
  - 保存模型状态字典
  - 加载和恢复训练

#### 1.9 GPU 训练
- **使用 GPU 优化训练性能**
  - **GPU 设备上的 PyTorch 计算**
    - 设备选择（CPU/GPU）
    - 张量移动到 GPU
    - GPU 内存管理
  - **单 GPU 训练**
    - 模型和数据迁移到 GPU
    - 单 GPU 训练流程
  - **多 GPU 训练**
    - DataParallel（数据并行）
    - DistributedDataParallel（分布式数据并行）
    - 多 GPU 训练策略和优化

**教程特色：**
- **快速入门** - 约一小时的阅读时间掌握 PyTorch 核心概念
- **实践导向** - 从基础到实际应用的完整代码示例
- **GPU 加速** - 详细的单 GPU 和多 GPU 训练指南
- **LLM 应用** - 特别关注大语言模型等实际应用场景
- **交互式学习** - Jupyter Notebook 格式，支持直接运行和实验

## 🎯 适用人群

- **AI 工程师** - 需要快速掌握 PyTorch 框架进行深度学习开发
- **深度学习研究者** - 研究神经网络训练、大模型开发、性能优化
- **算法开发者** - 实现和优化深度学习算法
- **学生** - 学习深度学习和 PyTorch 框架
- **数据科学家** - 使用 PyTorch 进行数据分析和建模

## 🔍 技术特色

### 理论与实践结合
- 从基础概念到实际应用
- 提供完整的代码示例
- Jupyter Notebook 交互式学习体验
- 可直接运行和实验

### 全面覆盖
- **基础层面** - 张量操作、自动微分、计算图
- **应用层面** - 神经网络实现、数据加载、训练循环
- **优化层面** - GPU 加速、多 GPU 训练、性能优化
- **实践层面** - 完整的训练流程、模型保存、最佳实践

### 快速上手
- 一小时快速入门
- 重点突出核心概念
- 适合新手学习
- 为实际项目开发打下基础

## 📚 学习路径建议

### 入门路径
1. **PyTorch 基础** → 理解 PyTorch 的三个核心组件
2. **张量操作** → 学习张量的创建、运算和 GPU 加速
3. **自动微分** → 理解 Autograd 机制和反向传播

### 进阶路径
1. **神经网络实现** → 学习如何构建和训练神经网络
2. **数据加载** → 掌握 DataLoader 和数据处理
3. **训练循环** → 实现完整的训练流程

### 高级路径
1. **GPU 训练** → 学习单 GPU 和多 GPU 训练
2. **性能优化** → 优化训练速度和内存使用
3. **实际应用** → 应用到实际项目和大模型训练

## 🔗 相关资源

### 官方文档
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [PyTorch 示例](https://github.com/pytorch/examples)

### 学习资源
- [PyTorch 一小时教程原文](https://sebastianraschka.com/teaching/pytorch-1h/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [深度学习入门](https://d2l.ai/)

### 相关目录
- [CUDA 编程](../CUDA/) - CUDA 核心概念与 GPU 编程
- [GPU 架构](../GPU/) - GPU 硬件架构与优化
- [Training 训练](../Training/) - 模型训练策略与优化

---

**注意**：本目录内容持续更新中。如有问题或建议，欢迎反馈。

