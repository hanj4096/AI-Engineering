# AI-Engineering

AI 工程学习资料库 - GPU 计算、CUDA 编程、PyTorch 深度学习与 GPU 管理技术深度解析

## 📚 项目简介

本项目是一个专注于 AI 工程领域的技术学习资料库，深入解析 GPU 计算架构、CUDA 编程实践、PyTorch 深度学习框架、GPU 管理技术等核心主题。内容涵盖从基础理论到实践应用的完整知识体系，适合 AI 工程师、系统架构师、高性能计算开发者等技术人员学习和参考。

## 🗂️ 目录结构

```
AI-Engineering/
├── CUDA/                    # CUDA 编程核心概念与实践
├── GPU/                     # GPU 相关技术文档
│   ├── gpu_architecture/    # GPU 架构深度解析
│   ├── gpu_programming/     # GPU 编程入门指南
│   └── gpu_manager/         # GPU 管理技术（虚拟化、切分、远程调用）
├── PyTorch/                 # PyTorch 深度学习框架教程
└── README.md                # 项目总览文档
```

## 📖 核心内容

### 1. CUDA 编程 (`CUDA/`)

CUDA（Compute Unified Device Architecture）并行计算平台和编程模型的深度解析。

**主要内容：**
- **CUDA 核心概念** - CUDA 架构层次、编程模型、内存模型
- **CUDA 核心详解** - NVIDIA CUDA 核心架构与并行处理原理
- **CUDA 流（Streams）** - 异步执行、流管理、性能优化
- **TileLang 快速入门** - 高级 GPU 编程语言实践
- **CUDA 编程实践** - 基础与实践 PDF 教程
- **Professional CUDA C Programming** - 专业 CUDA C 编程参考

**核心文档：**
- [CUDA 核心概念简介](CUDA/README.md)
- [深入了解 NVIDIA CUDA 核心](CUDA/cuda_cores_cn.md)
- [CUDA 流详细介绍](CUDA/cuda_streams.md)
- [TileLang 快速入门](CUDA/TileLang_快速入门.md)
- [CUDA 编程简介 - 基础与实践](CUDA/CUDA%20编程简介%20-%20基础与实践.pdf)

### 2. GPU 架构 (`GPU/gpu_architecture/`)

GPU 硬件架构的深度技术解析，从基础概念到具体硬件实例的全面分析。

**主要内容：**
- **GPU 特性分析** - GPU vs CPU 架构对比、并行计算原理
- **GPU 内存系统** - 内存层次结构、带宽优化、访问模式
- **硬件实例分析** - Tesla V100、RTX 5000 等具体 GPU 架构解析
- **GPGPU vs NPU** - 大模型推理与训练的算力选择指南
- **实践练习** - 设备查询、带宽测试等动手实践

**核心文档：**
- [GPU 架构深度解析](GPU/gpu_architecture/README.md)
- [GPU 特性分析](GPU/gpu_architecture/gpu_characteristics.md)
- [GPU 内存层次结构详解](GPU/gpu_architecture/gpu_memory.md)
- [GPGPU vs NPU：大模型推理与训练对比](GPU/gpu_architecture/GPGPU_vs_NPU_大模型推理训练对比.md)
- [Tesla V100 架构分析](GPU/gpu_architecture/tesla_v100.md)
- [RTX 5000 架构特性](GPU/gpu_architecture/rtx_5000.md)
- [设备查询练习](GPU/gpu_architecture/exer_device_query.md)
- [内存带宽测试练习](GPU/gpu_architecture/exer_device_bandwidth.md)

### 3. GPU 编程 (`GPU/gpu_programming/`)

GPU 架构和编程的基础理论指导，为 GPU 编程学习和实践提供入门指南。

**主要内容：**
- **GPU 架构基础** - GPU 硬件架构和设计原理
- **并行计算概念** - GPU 并行计算核心概念
- **CUDA 编程实践** - CUDA 并行编程模型
- **性能优化技巧** - GPU 程序性能优化方法
- **调试技术** - GPU 程序调试和分析

**核心文档：**
- [GPU 编程入门指南](GPU/gpu_programming/README.md)
- [GPU 架构与编程简介](GPU/gpu_programming/gpu_programming_introduction.md)

### 4. GPU 管理 (`GPU/gpu_manager/`)

GPU 虚拟化、切分、远程调用等核心技术的深度解析和实现代码。

**主要内容：**
- **GPU 虚拟化技术** - 硬件级、内核级、用户空间虚拟化
- **GPU 切分技术** - 时间切分、空间切分、混合切分策略
- **远程 GPU 调用** - 网络协议、数据传输、延迟优化
- **容器化 GPU 管理** - NVIDIA Container Toolkit、OCI 运行时集成
- **Kubernetes GPU 编排** - Device Plugin、资源调度、MIG 支持
- **实践代码** - 完整的实现代码和配置文件

**核心文档：**
- [GPU 管理技术深度解析](GPU/gpu_manager/README.md)
- [GPU 管理相关技术深度解析 - 虚拟化、切分及远程调用](GPU/gpu_manager/GPU%20管理相关技术深度解析%20-%20虚拟化、切分及远程调用.md)
- [GPU 虚拟化与切分技术原理解析](GPU/gpu_manager/GPU虚拟化与切分技术原理解析.md)
- [第一部分：基础理论篇](GPU/gpu_manager/第一部分：基础理论篇.md)
- [第二部分：虚拟化技术篇](GPU/gpu_manager/第二部分：虚拟化技术篇.md)
- [第三部分：资源管理与优化篇](GPU/gpu_manager/第三部分：资源管理与优化篇.md)
- [第四部分：实践应用篇](GPU/gpu_manager/第四部分：实践应用篇.md)

**代码实现：**
- [GPU 管理代码库](GPU/gpu_manager/code/) - 完整的 GPU 虚拟化、切分、调度等核心模块实现
- [配置文件集合](GPU/gpu_manager/configs/) - 云平台、容器化、监控等配置文件

### 5. PyTorch 深度学习 (`PyTorch/`)

PyTorch 深度学习框架的快速入门教程，从张量基础到多 GPU 训练的完整学习路径。

**主要内容：**
- **PyTorch 核心组件** - 张量库、自动微分引擎、深度学习库
- **张量操作** - 张量数据类型、常见操作、GPU 加速
- **自动微分** - Autograd 机制、反向传播、梯度计算
- **神经网络实现** - 多层神经网络、损失函数、优化器
- **数据加载** - DataLoader、数据预处理、批处理
- **训练循环** - 典型的训练流程、验证、模型保存
- **GPU 训练** - 单 GPU 训练、多 GPU 训练、性能优化

**核心文档：**
- [PyTorch 一小时教程：从张量到在多GPU上训练神经网络](PyTorch/pytorch-1h.ipynb)

**教程特色：**
- **快速入门** - 约一小时的阅读时间掌握 PyTorch 核心概念
- **实践导向** - 从基础到实际应用的完整代码示例
- **GPU 加速** - 详细的单 GPU 和多 GPU 训练指南
- **LLM 应用** - 特别关注大语言模型等实际应用场景

## 🎯 适用人群

- **AI 工程师** - 需要深入理解 GPU 计算原理、PyTorch 框架和优化技巧
- **深度学习研究者** - 研究神经网络训练、大模型开发、性能优化
- **系统架构师** - 设计 GPU 资源管理和调度系统
- **高性能计算开发者** - 开发 GPU 加速的并行计算应用
- **DevOps 工程师** - 部署和管理 GPU 集群、容器化 GPU 应用
- **研究人员** - 研究 GPU 架构、并行计算、AI 加速技术

## 🔍 技术特色

### 理论与实践结合
- 从基础概念到实际硬件分析
- 提供可执行的代码示例和练习
- 结合具体 GPU 型号的详细分析
- Jupyter Notebook 交互式学习体验

### 全面覆盖
- **硬件层面** - GPU 架构、内存系统、计算单元
- **软件层面** - CUDA 编程、PyTorch 深度学习、并行算法、性能优化
- **系统层面** - 虚拟化、资源管理、容器化部署、Kubernetes 编排

### 深度解析
- 技术原理的深入剖析
- 性能特征与优化策略
- 实际应用场景与最佳实践
- 从入门到高级的完整学习路径

## 📚 学习路径建议

### 入门路径
1. **PyTorch 基础** → `PyTorch/pytorch-1h.ipynb` - 快速掌握深度学习框架
2. **GPU 编程基础** → `GPU/gpu_programming/`
3. **CUDA 核心概念** → `CUDA/README.md`
4. **GPU 架构基础** → `GPU/gpu_architecture/gpu_characteristics.md`

### 进阶路径
1. **PyTorch 深度学习实践** → `PyTorch/pytorch-1h.ipynb` 完整学习
2. **CUDA 编程实践** → `CUDA/` 完整学习
3. **GPU 内存优化** → `GPU/gpu_architecture/gpu_memory.md`
4. **硬件实例分析** → `GPU/gpu_architecture/tesla_v100.md`

### 高级路径
1. **GPU 虚拟化技术** → `GPU/gpu_manager/` 完整学习
2. **容器化与编排** → `GPU/gpu_manager/` 相关章节
3. **性能优化实践** → 各目录的实践练习和代码实现
4. **多 GPU 训练优化** → `PyTorch/pytorch-1h.ipynb` 多 GPU 章节

## 🔗 相关资源

### 官方文档
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Kubernetes Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)

### 学习资源
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [PyTorch 示例](https://github.com/pytorch/examples)

### 社区支持
- [NVIDIA 开发者论坛](https://forums.developer.nvidia.com/)
- [PyTorch 论坛](https://discuss.pytorch.org/)
- [Stack Overflow - CUDA](https://stackoverflow.com/questions/tagged/cuda)
- [Stack Overflow - PyTorch](https://stackoverflow.com/questions/tagged/pytorch)

## 📄 许可证

本项目采用 [LICENSE](LICENSE) 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来完善本项目。

---

**注意**：本项目为学习资料库，内容持续更新中。如有问题或建议，欢迎反馈。
