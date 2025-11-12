# GPU - GPU 相关技术

GPU（Graphics Processing Unit）相关技术的综合学习资料库，涵盖 GPU 架构、GPU 编程、GPU 管理等核心主题。

## 📚 项目简介

本目录专注于 GPU 技术的全面学习与实践，深入解析 GPU 硬件架构、编程方法、资源管理等核心技术。内容涵盖从基础理论到实际应用的完整知识体系，适合 AI 工程师、GPU 开发者、系统架构师、高性能计算开发者等技术人员学习和参考。

## 🗂️ 目录结构

```
GPU/
├── gpu_architecture/        # GPU 架构深度解析
│   ├── README.md            # GPU 架构总览
│   ├── gpu_characteristics.md    # GPU 特性分析
│   ├── gpu_memory.md        # GPU 内存层次结构详解
│   ├── tesla_v100.md        # Tesla V100 架构分析
│   ├── rtx_5000.md          # RTX 5000 架构特性
│   ├── GPGPU_vs_NPU_大模型推理训练对比.md
│   ├── exer_device_query.md     # 设备查询练习
│   └── exer_device_bandwidth.md # 内存带宽测试练习
├── gpu_programming/         # GPU 编程入门指南
│   ├── README.md            # GPU 编程总览
│   └── TileLang_快速入门.md  # TileLang 高级 GPU 编程语言快速入门
└── gpu_manager/             # GPU 管理技术（虚拟化、切分、远程调用）
    ├── README.md            # GPU 管理技术总览
    ├── GPU 管理相关技术深度解析 - 虚拟化、切分及远程调用.md
    ├── GPU虚拟化与切分技术原理解析.md
    ├── 第一部分：基础理论篇.md
    ├── 第二部分：虚拟化技术篇.md
    ├── 第三部分：资源管理与优化篇.md
    ├── 第四部分：实践应用篇.md
    ├── code/                # 完整实现代码
    └── configs/             # 配置文件集合
```

## 📖 核心内容

### 1. GPU 架构 (`gpu_architecture/`)

GPU 硬件架构的深度技术解析，从基础概念到具体硬件实例的全面分析。

**主要内容：**
- **GPU 特性分析** - GPU vs CPU 架构对比、并行计算原理
- **GPU 内存系统** - 内存层次结构、带宽优化、访问模式
- **硬件实例分析** - Tesla V100、RTX 5000 等具体 GPU 架构解析
- **GPGPU vs NPU** - 大模型推理与训练的算力选择指南
- **实践练习** - 设备查询、带宽测试等动手实践

**核心文档：**
- [GPU 架构深度解析](gpu_architecture/README.md)
- [GPU 特性分析](gpu_architecture/gpu_characteristics.md)
- [GPU 内存层次结构详解](gpu_architecture/gpu_memory.md)
- [GPGPU vs NPU：大模型推理与训练对比](gpu_architecture/GPGPU_vs_NPU_大模型推理训练对比.md)
- [Tesla V100 架构分析](gpu_architecture/tesla_v100.md)
- [RTX 5000 架构特性](gpu_architecture/rtx_5000.md)
- [设备查询练习](gpu_architecture/exer_device_query.md)
- [内存带宽测试练习](gpu_architecture/exer_device_bandwidth.md)

**技术特色：**
- 从架构原理到实际硬件分析
- GPU 与 CPU 架构深度对比
- 基于具体 GPU 型号的详细分析
- 提供可执行的测试和分析练习

### 2. GPU 编程 (`gpu_programming/`)

GPU 架构和编程的基础理论指导，为 GPU 编程学习和实践提供入门指南。

**主要内容：**
- **GPU 架构基础** - GPU 硬件架构和设计原理
- **并行计算概念** - GPU 并行计算核心概念
- **CUDA 编程实践** - CUDA 并行编程模型
- **性能优化技巧** - GPU 程序性能优化方法
- **调试技术** - GPU 程序调试和分析

**核心文档：**
- [TileLang 快速入门](gpu_programming/TileLang_快速入门.md) - 高级 GPU 编程语言实践指南

**注意**：GPU 架构与编程入门文档位于 `../CUDA/gpu_programming_introduction.md`，请参考 CUDA 目录。

**技术特色：**
- 深入理解 GPU 的硬件架构和设计原理
- 掌握 GPU 并行计算的核心概念
- 了解 GPU 内存系统的层次结构
- CUDA 并行编程模型和实践
- TileLang 高级 GPU 编程语言

**TileLang 特色：**
- **简洁的语法** - 采用 Python 风格的语法，降低学习成本
- **高性能** - 基于 TVM 编译器基础设施，提供底层优化能力
- **跨平台支持** - 支持 NVIDIA GPU（CUDA）、AMD GPU（HIP）和 CPU
- **丰富的算子支持** - 内置 GEMM、FlashAttention、LinearAttention 等高性能算子
- **自动优化** - 支持自动流水线、内存布局优化、L2 缓存友好的重排等特性

**TileLang 适用场景：**
- 高性能矩阵运算（GEMM、量化 GEMM）
- 注意力机制实现（FlashAttention、LinearAttention）
- 自定义 GPU 内核开发
- 深度学习算子优化

### 3. GPU 管理 (`gpu_manager/`)

GPU 虚拟化、切分、远程调用等核心技术的深度解析和实现代码。

**主要内容：**
- **GPU 虚拟化技术** - 硬件级、内核级、用户空间虚拟化
- **GPU 切分技术** - 时间切分、空间切分、混合切分策略
- **远程 GPU 调用** - 网络协议、数据传输、延迟优化
- **容器化 GPU 管理** - NVIDIA Container Toolkit、OCI 运行时集成
- **Kubernetes GPU 编排** - Device Plugin、资源调度、MIG 支持
- **实践代码** - 完整的实现代码和配置文件

**核心文档：**
- [GPU 管理技术深度解析](gpu_manager/README.md)
- [GPU 管理相关技术深度解析 - 虚拟化、切分及远程调用](gpu_manager/GPU%20管理相关技术深度解析%20-%20虚拟化、切分及远程调用.md)
- [GPU 虚拟化与切分技术原理解析](gpu_manager/GPU虚拟化与切分技术原理解析.md)
- [第一部分：基础理论篇](gpu_manager/第一部分：基础理论篇.md)
- [第二部分：虚拟化技术篇](gpu_manager/第二部分：虚拟化技术篇.md)
- [第三部分：资源管理与优化篇](gpu_manager/第三部分：资源管理与优化篇.md)
- [第四部分：实践应用篇](gpu_manager/第四部分：实践应用篇.md)

**代码实现：**
- [GPU 管理代码库](gpu_manager/code/) - 完整的 GPU 虚拟化、切分、调度等核心模块实现
- [配置文件集合](gpu_manager/configs/) - 云平台、容器化、监控等配置文件

**核心技术特性：**
- **GPU 虚拟化技术** - SR-IOV、MIG、内核拦截、API 层虚拟化
- **GPU 切分技术** - 时间片轮转、物理资源分割、混合策略
- **远程 GPU 调用** - 高效网络协议、数据传输优化、低延迟技术
- **容器化 GPU 管理** - Container Toolkit、OCI 集成、CDI 规范、设备隔离
- **Kubernetes GPU 编排** - Device Plugin、资源调度、MIG 支持、健康监控

## 🎯 适用人群

- **AI 工程师** - 需要深入理解 GPU 计算原理和优化技巧
- **GPU 开发者** - 开发 GPU 加速应用、优化 GPU 性能
- **系统架构师** - 设计 GPU 资源管理和调度系统
- **高性能计算开发者** - 开发 GPU 加速的并行计算应用
- **DevOps 工程师** - 部署和管理 GPU 集群、容器化 GPU 应用
- **研究人员** - 研究 GPU 架构、并行计算、AI 加速技术

## 🔍 技术特色

### 理论与实践结合
- 从基础概念到实际硬件分析
- 提供可执行的代码示例和练习
- 结合具体 GPU 型号的详细分析
- 完整的实现代码和配置文件

### 全面覆盖
- **硬件层面** - GPU 架构、内存系统、计算单元
- **软件层面** - GPU 编程、并行算法、性能优化
- **系统层面** - 虚拟化、资源管理、容器化部署、Kubernetes 编排

### 深度解析
- 技术原理的深入剖析
- 性能特征与优化策略
- 实际应用场景与最佳实践
- 从入门到高级的完整学习路径

## 📚 学习路径建议

### 入门路径
1. **GPU 架构基础** → `gpu_architecture/gpu_characteristics.md` - 理解 GPU 基本特性
2. **GPU 编程基础** → `../CUDA/gpu_programming_introduction.md` - 学习 GPU 编程入门
3. **TileLang 入门** → `gpu_programming/TileLang_快速入门.md` - 学习高级 GPU 编程语言
4. **设备查询练习** → `gpu_architecture/exer_device_query.md` - 动手实践

### 进阶路径
1. **GPU 内存系统** → `gpu_architecture/gpu_memory.md` - 深入理解内存层次
2. **硬件实例分析** → `gpu_architecture/tesla_v100.md` - 学习具体 GPU 架构
3. **内存带宽测试** → `gpu_architecture/exer_device_bandwidth.md` - 性能测试实践

### 高级路径
1. **GPU 虚拟化技术** → `gpu_manager/` 完整学习
2. **容器化与编排** → `gpu_manager/` 相关章节
3. **性能优化实践** → 各目录的实践练习和代码实现

## 🔗 相关资源

### 官方文档
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Kubernetes Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)

### 学习资源
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU 架构白皮书](https://www.nvidia.com/en-us/data-center/resources/architecture-whitepapers/)

### 相关目录
- [CUDA 编程](../CUDA/) - CUDA 核心概念与编程实践
- [PyTorch 深度学习](../PyTorch/) - PyTorch 框架与 GPU 训练

---

**注意**：本目录内容持续更新中。如有问题或建议，欢迎反馈。

