# AI-Engineering

AI 工程学习资料库 - 从 GPU 计算到 LLM 应用的完整技术栈深度解析

## 📚 项目简介

本项目是一个专注于 AI 工程领域的技术学习资料库，全面覆盖 GPU 计算架构、CUDA 编程实践、PyTorch 深度学习框架、大语言模型应用、MLOps 工程实践等核心主题。内容涵盖从基础理论到生产级应用的完整知识体系，适合 AI 工程师、系统架构师、高性能计算开发者、LLM 应用开发者等技术人员学习和参考。

## 🗂️ 目录结构

```
AI-Engineering/
├── agent/                      # AI 智能体技术
├── context-engineering/        # 上下文工程
├── cuda/                       # CUDA 编程核心概念与实践
├── gpu/                        # GPU 相关技术文档
│   ├── gpu-arch/               # GPU 架构深度解析
│   ├── gpu-programming/        # GPU 编程入门指南
│   └── gpu-manager/            # GPU 管理技术（虚拟化、切分、远程调用）
├── inference/                  # 模型推理技术
├── math/                       # AI 数学基础
├── mlops/                      # 机器学习运维
├── prompt/                     # 提示工程
├── pytorch/                    # PyTorch 深度学习框架教程
├── qwen/                       # 通义千问大语言模型
├── rag/                        # 检索增强生成
├── reinforcement-learning/     # 强化学习
├── training/                   # 模型训练
├── triton/                     # Triton GPU 编程语言
├── xai/                        # 可解释 AI
└── README.md                   # 项目总览文档
```

## 📖 核心内容

### 1. CUDA 编程 (`cuda/`)

CUDA（Compute Unified Device Architecture）并行计算平台和编程模型的深度解析。

**主要内容：**
- **CUDA 核心概念** - CUDA 架构层次、编程模型、内存模型
- **CUDA 核心详解** - NVIDIA CUDA 核心架构与并行处理原理
- **CUDA 流（Streams）** - 异步执行、流管理、性能优化
- **CUDA 编程实践** - 基础与实践 PDF 教程
- **Professional CUDA C Programming** - 专业 CUDA C 编程参考

**核心文档：**
- [CUDA 核心概念简介](cuda/README.md)
- [深入了解 NVIDIA CUDA 核心](cuda/cuda-cores.md)
- [CUDA 流详细介绍](cuda/cuda-streams.md)
- [CUDA 编程简介 - 基础与实践](cuda/cuda-programming-intro.pdf)

### 2. GPU 架构 (`gpu/gpu-arch/`)

GPU 硬件架构的深度技术解析，从基础概念到具体硬件实例的全面分析。

**主要内容：**
- **GPU 特性分析** - GPU vs CPU 架构对比、并行计算原理
- **GPU 内存系统** - 内存层次结构、带宽优化、访问模式
- **硬件实例分析** - Tesla V100、RTX 5000 等具体 GPU 架构解析
- **GPGPU vs NPU** - 大模型推理与训练的算力选择指南
- **实践练习** - 设备查询、带宽测试等动手实践

**核心文档：**
- [GPU 架构深度解析](gpu/gpu-arch/README.md)
- [GPU 特性分析](gpu/gpu-arch/gpu_characteristics.md)
- [GPU 内存层次结构详解](gpu/gpu_architecture/gpu_memory.md)
- [GPGPU vs NPU：大模型推理与训练对比](gpu/gpu-arch/GPGPU_vs_NPU_大模型推理训练对比.md)
- [Tesla V100 架构分析](gpu/gpu-arch/tesla_v100.md)
- [RTX 5000 架构特性](gpu/gpu-arch/rtx_5000.md)

### 3. GPU 编程 (`gpu/gpu_programming/`)

GPU 架构和编程的基础理论指导，为 GPU 编程学习和实践提供入门指南。

**主要内容：**
- **GPU 架构基础** - GPU 硬件架构和设计原理
- **并行计算概念** - GPU 并行计算核心概念
- **CUDA 编程实践** - CUDA 并行编程模型
- **性能优化技巧** - GPU 程序性能优化方法
- **调试技术** - GPU 程序调试和分析

**核心文档：**
- [GPU 编程入门指南](gpu/gpu-programming/README.md)
- [GPU 架构与编程简介](gpu/gpu-programming/gpu_programming_introduction.md)

### 4. GPU 管理 (`gpu/gpu-manager/`)

GPU 虚拟化、切分、远程调用等核心技术的深度解析和实现代码。

**主要内容：**
- **GPU 虚拟化技术** - 硬件级、内核级、用户空间虚拟化
- **GPU 切分技术** - 时间切分、空间切分、混合切分策略
- **远程 GPU 调用** - 网络协议、数据传输、延迟优化
- **容器化 GPU 管理** - NVIDIA Container Toolkit、OCI 运行时集成
- **Kubernetes GPU 编排** - Device Plugin、资源调度、MIG 支持
- **实践代码** - 完整的实现代码和配置文件

**核心文档：**
- [GPU 管理技术深度解析](gpu/gpu-manager/README.md)
- [GPU 管理相关技术深度解析 - 虚拟化、切分及远程调用](gpu/gpu-manager/GPU%20管理相关技术深度解析%20-%20虚拟化、切分及远程调用.md)
- [GPU 虚拟化与切分技术原理解析](gpu/gpu-manager/GPU虚拟化与切分技术原理解析.md)
- [第一部分：基础理论篇](gpu/gpu-manager/第一部分：基础理论篇.md)
- [第二部分：虚拟化技术篇](gpu/gpu-manager/第二部分：虚拟化技术篇.md)
- [第三部分：资源管理与优化篇](gpu/gpu-manager/第三部分：资源管理与优化篇.md)
- [第四部分：实践应用篇](gpu/gpu-manager/第四部分：实践应用篇.md)

**代码实现：**
- [GPU 管理代码库](gpu/gpu-manager/code/) - 完整的 GPU 虚拟化、切分、调度等核心模块实现
- [配置文件集合](gpu/gpu-manager/configs/) - 云平台、容器化、监控等配置文件

### 5. PyTorch 深度学习 (`pytorch/`)

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
- [PyTorch 一小时教程：从张量到在多GPU上训练神经网络](pytorch/pytorch-1h.ipynb)

**教程特色：**
- **快速入门** - 约一小时的阅读时间掌握 PyTorch 核心概念
- **实践导向** - 从基础到实际应用的完整代码示例
- **GPU 加速** - 详细的单 GPU 和多 GPU 训练指南
- **LLM 应用** - 特别关注大语言模型等实际应用场景

### 6. Triton GPU 编程 (`triton/`)

Triton 是一种用于编写高效 GPU 内核的 Python 语言和编译器，由 OpenAI 开发。

**主要内容：**
- **Triton 基础** - Triton 简介、安装配置、语法特性
- **内存管理** - 内存层次、访问模式、内存优化
- **性能优化** - 自动优化、手动优化、性能分析
- **实际应用** - 矩阵运算、注意力机制、自定义算子
- **高级特性** - 自动调优、多 GPU 支持、混合精度
- **PyTorch 集成** - 与 PyTorch 的集成使用、自定义算子注册

**核心文档：**
- [Triton GPU 编程语言](triton/README.md)

### 7. 模型训练 (`training/`)

模型训练技术的学习资料库，涵盖训练策略、优化算法、分布式训练、大模型训练等核心主题。

**主要内容：**
- **训练基础** - 训练流程、损失函数、优化器、学习率调度
- **训练策略** - 数据策略、批处理策略、正则化、早停策略
- **优化算法** - 一阶优化、二阶优化、自适应优化、优化技巧
- **分布式训练** - 数据并行、模型并行、分布式框架、通信优化
- **大模型训练** - 预训练、微调技术、训练稳定性、内存优化
- **训练加速** - 混合精度训练、编译优化、硬件加速、流水线优化
- **训练监控** - 训练指标、可视化工具、实验跟踪、异常检测
- **超参数调优** - 网格搜索、随机搜索、自动调优、调优策略

**核心文档：**
- [模型训练技术](training/README.md)

### 8. 模型推理 (`inference/`)

模型推理技术的学习资料库，涵盖推理优化、部署策略、性能调优、量化加速等核心主题。

**主要内容：**
- **推理基础** - 推理流程、推理引擎、模型格式、推理框架
- **推理优化技术** - 图优化、算子优化、内存优化、批处理优化
- **量化与压缩** - INT8/FP16 量化、量化感知训练、模型压缩、量化推理
- **硬件加速** - GPU 推理、CPU 优化、专用芯片、混合部署
- **部署与运维** - 服务化部署、容器化部署、边缘部署、监控运维
- **性能调优** - 延迟优化、吞吐量优化、资源优化、基准测试

**核心文档：**
- [模型推理技术](inference/README.md)

### 9. MLOps (`mlops/`)

机器学习运维技术的学习资料库，涵盖模型部署、持续集成、监控运维、自动化流水线等核心主题。

**主要内容：**
- **MLOps 基础** - MLOps 概念、MLOps vs DevOps、成熟度模型、工具链
- **模型版本管理** - 模型注册、版本控制、实验跟踪、模型元数据
- **CI/CD** - CI/CD 流水线、模型测试、自动化部署、回滚策略
- **模型监控** - 性能监控、数据监控、模型监控、告警系统
- **数据管理** - 数据版本管理、特征存储、数据质量、数据管道
- **模型服务化** - 模型服务、服务框架、服务编排、边缘部署
- **自动化与编排** - 工作流编排、自动化训练、资源管理、多环境管理
- **安全与合规** - 模型安全、数据隐私、合规要求、可解释性

**核心文档：**
- [MLOps 机器学习运维](mlops/README.md)

### 10. RAG 检索增强生成 (`rag/`)

检索增强生成技术的学习资料库，涵盖 RAG 架构、向量数据库、检索策略、应用实践等核心主题。

**主要内容：**
- **RAG 基础** - RAG 概念、RAG 架构、RAG vs 微调、RAG 优势
- **文档处理** - 文档加载、文档分块、文档预处理、多模态文档
- **向量化与嵌入** - 嵌入模型、向量化策略、嵌入优化、向量质量
- **向量数据库** - 数据库选型、索引技术、检索优化、数据库管理
- **检索策略** - 检索方法、重排序、多跳检索、检索优化
- **生成与融合** - 提示构建、生成策略、结果融合、后处理
- **高级 RAG 技术** - Self-RAG、Corrective RAG、GraphRAG、Multi-Agent RAG
- **应用实践** - 知识问答、文档分析、代码助手、研究助手
- **性能优化** - 检索优化、生成优化、系统优化、成本优化
- **评估与监控** - 评估指标、评估方法、监控系统、持续改进

**核心文档：**
- [RAG 检索增强生成](rag/README.md)

### 11. AI 智能体 (`agent/`)

AI 智能体技术的学习资料库，涵盖智能体架构、多智能体系统、工具调用、自主决策等核心主题。

**主要内容：**
- **智能体基础** - 智能体架构、智能体类型、感知与行动、决策机制
- **多智能体系统** - 多智能体协作、竞争与博弈、分布式智能、智能体组织
- **工具调用与集成** - 工具调用框架、外部工具集成、工具链设计、工具学习
- **自主决策与规划** - 规划算法、强化学习应用、推理机制、长期规划
- **实际应用场景** - 代码智能体、数据分析智能体、对话智能体、自动化智能体

**核心文档：**
- [AI 智能体技术](agent/README.md)

**智能体设计模式** (`agent/agentic-design-patterns/`)：

《Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems》的中文翻译（内容有修改），提供21个核心设计模式的完整指南。

**核心设计模式：**
- 提示链、路由、并行化、反思、工具使用、规划、多智能体协作、记忆管理、适应、知识检索、异常处理、人机交互、目标设定、资源优化、推理技术、安全模式、评估监控、优先级、探索发现等

**Notebook 代码示例：**
- 提供各章节的 Jupyter Notebook 示例（已配置 Qwen 模型）
- 技术栈：LangChain、LangGraph、Google ADK、CrewAI + Qwen 模型
- 快速开始：`agent/agentic-design-patterns/README.md`

### 12. 提示工程 (`prompt/`)

提示工程技术的学习资料库，涵盖提示设计、优化策略、Few-shot 学习、Chain-of-Thought 等核心主题。

**主要内容：**
- **提示工程基础** - 提示工程概念、提示设计原则、提示类型、提示模板
- **基础提示技巧** - 明确指令、角色设定、示例引导、输出控制
- **高级提示技术** - Chain-of-Thought、Zero-shot CoT、Self-Consistency、Tree of Thoughts
- **提示优化策略** - 提示迭代、A/B 测试、提示压缩、提示组合
- **领域特定提示** - 代码生成、数据分析、文本处理、创意写作
- **提示工程工具** - 提示管理、提示测试、提示优化、提示分析
- **多模态提示** - 图像理解、多模态生成、提示设计
- **提示安全与伦理** - 提示注入、偏见控制、内容安全、隐私保护

**核心文档：**
- [提示工程](prompt/README.md)

### 13. 上下文工程 (`context-engineering/`)

上下文工程技术的学习资料库，涵盖上下文管理、长上下文处理、上下文优化、应用实践等核心主题。

**主要内容：**
- **上下文工程基础** - 上下文概念、上下文类型、上下文管理、上下文限制
- **上下文构建** - 上下文设计、上下文注入、上下文模板、上下文优化
- **长上下文处理** - 长文本处理、上下文压缩、上下文检索、上下文融合
- **多轮对话上下文** - 对话历史管理、上下文窗口、上下文压缩、上下文恢复
- **上下文优化技术** - 信息优先级、上下文压缩、上下文选择、上下文更新
- **上下文工程工具** - 上下文管理库、上下文分析、上下文测试、上下文监控
- **高级上下文技术** - 分层上下文、动态上下文、多模态上下文、上下文推理
- **应用场景** - 长文档处理、多轮对话、代码生成、知识问答
- **性能优化** - Token 优化、延迟优化、成本优化、质量优化
- **最佳实践** - 上下文设计原则、上下文管理策略、性能优化策略、错误处理

**核心文档：**
- [上下文工程](context-engineering/README.md)

### 14. 通义千问 (`qwen/`)

通义千问大语言模型的学习资料库，涵盖模型架构、使用方法、微调实践、应用开发等核心主题。

**主要内容：**
- **Qwen 模型概览** - 模型系列、模型规模、模型特点、模型对比
- **模型使用** - 快速开始、API 调用、本地部署、对话交互
- **模型微调** - 微调方法、微调框架、数据准备、训练技巧
- **应用开发** - RAG 应用、Agent 应用、代码生成、多模态应用
- **性能优化** - 推理优化、加速技术、硬件优化、成本优化
- **部署与运维** - 服务化部署、高可用部署、监控运维、安全部署
- **最佳实践** - 使用最佳实践、微调最佳实践、部署最佳实践、应用最佳实践

**核心文档：**
- [通义千问](qwen/README.md)

### 15. 强化学习 (`reinforcement-learning/`)

强化学习技术的学习资料库，涵盖基础理论、经典算法、深度强化学习、应用实践等核心主题。

**主要内容：**
- **强化学习基础** - 基本概念、马尔可夫决策过程、价值函数、最优策略
- **经典算法** - 动态规划、蒙特卡洛方法、时序差分学习、函数近似
- **深度强化学习** - DQN、策略梯度、PPO、SAC
- **高级算法** - 多智能体强化学习、分层强化学习、模仿学习、元强化学习
- **应用领域** - 游戏 AI、机器人控制、自动驾驶、推荐系统
- **强化学习框架** - Gym/Gymnasium、Stable-Baselines3、Ray RLlib、Tianshou
- **训练技巧** - 超参数调优、稳定性技巧、样本效率、收敛加速
- **评估与分析** - 性能评估、算法分析、对比实验、理论分析

**核心文档：**
- [强化学习](reinforcement-learning/README.md)

### 16. AI 数学基础 (`math/`)

AI 数学基础的学习资料库，涵盖线性代数、概率论、微积分、优化理论等核心数学知识。

**主要内容：**
- **线性代数** - 向量与矩阵、特征值与特征向量、矩阵分解、应用场景
- **概率论与统计** - 概率基础、常见分布、统计推断、应用场景
- **微积分** - 导数与梯度、链式法则、积分、应用场景
- **优化理论** - 优化基础、优化算法、随机优化、应用场景
- **信息论** - 信息熵、互信息、编码理论、应用场景
- **图论** - 图基础、图算法、图神经网络、应用场景

**核心文档：**
- [AI 数学基础](math/README.md)

### 17. 可解释 AI (`xai/`)

可解释 AI（Explainable AI）技术的学习资料库，涵盖模型可解释性、解释方法、公平性分析、可解释性实践等核心主题。

**主要内容：**
- **白盒模型** - 线性回归、决策树、广义加性模型（GAM）、B-样条
- **模型无关方法 - 全局可解释性** - 树集成、部分依赖图（PDP）、特征交互
- **模型无关方法 - 局部可解释性** - LIME、SHAP、Anchors、深度神经网络解释
- **显著性映射** - 卷积神经网络可视化、反向传播方法、积分梯度、SmoothGrad、Grad-CAM
- **网络层和单元理解** - 网络解剖（Network Dissection）、层可视化、单元分析
- **语义相似性理解** - 主成分分析（PCA）、t-SNE、词嵌入可视化、语义分析
- **公平性和偏见缓解** - 公平性分析、偏见检测、公平性指标、偏见缓解方法
- **可解释 AI 路径** - 反事实解释、可解释性最佳实践、可解释性评估

**核心文档：**
- [可解释 AI](xai/README.md)
- [Interpretable AI Book](xai/interpretable-ai-book/README.md) - 可解释机器学习系统构建完整教程

**教程特色：**
- **完整代码实现** - 基于《Interpretable AI》书籍的 Jupyter Notebook 实现
- **实践导向** - 从白盒模型到黑盒模型解释的完整实践路径
- **多方法覆盖** - 涵盖全局解释、局部解释、可视化解释等多种方法
- **公平性分析** - 包含模型公平性分析和偏见缓解实践

## 🎯 适用人群

- **AI 工程师** - 需要深入理解 GPU 计算原理、深度学习框架、LLM 应用和优化技巧
- **深度学习研究者** - 研究神经网络训练、大模型开发、性能优化
- **系统架构师** - 设计 GPU 资源管理、MLOps 系统、推理服务架构
- **高性能计算开发者** - 开发 GPU 加速的并行计算应用
- **DevOps 工程师** - 部署和管理 GPU 集群、容器化 GPU 应用、MLOps 平台
- **LLM 应用开发者** - 开发 RAG、Agent、提示工程等 LLM 应用
- **智能体开发者** - 学习智能体设计模式、开发多智能体系统
- **可解释 AI 研究者** - 研究模型可解释性、公平性分析、解释方法
- **研究人员** - 研究 GPU 架构、并行计算、AI 加速技术、LLM 技术

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
- **应用层面** - 模型训练、推理优化、LLM 应用、MLOps 实践

### 深度解析
- 技术原理的深入剖析
- 性能特征与优化策略
- 实际应用场景与最佳实践
- 从入门到高级的完整学习路径

### LLM 应用全栈
- **基础技术** - 提示工程、上下文工程、RAG、Agent
- **智能体设计模式** - 21个核心设计模式的完整指南与实践代码
- **模型实践** - Qwen 模型使用、微调、部署（Notebook 示例已配置）
- **工程实践** - 训练、推理、MLOps、性能优化
- **数学基础** - AI 所需的数学知识体系

### 可解释性与公平性
- **可解释性方法** - 白盒模型、模型无关解释、显著性映射、网络解剖
- **公平性分析** - 偏见检测、公平性指标、偏见缓解
- **实践工具** - LIME、SHAP、Grad-CAM 等解释工具实践

## 📚 学习路径建议

### 入门路径
1. **PyTorch 基础** → `pytorch/pytorch-1h.ipynb` - 快速掌握深度学习框架
2. **GPU 编程基础** → `gpu/gpu-programming/`
3. **CUDA 核心概念** → `cuda/README.md`
4. **GPU 架构基础** → `gpu/gpu-arch/gpu_characteristics.md`
5. **AI 数学基础** → `math/README.md` - 理解 AI 算法的数学原理

### 进阶路径
1. **PyTorch 深度学习实践** → `pytorch/pytorch-1h.ipynb` 完整学习
2. **CUDA 编程实践** → `cuda/` 完整学习
3. **模型训练** → `training/README.md` - 学习训练策略和优化
4. **模型推理** → `inference/README.md` - 学习推理优化和部署
5. **GPU 内存优化** → `gpu/gpu-arch/gpu_memory.md`
6. **硬件实例分析** → `gpu/gpu-arch/tesla_v100.md`

### 高级路径
1. **GPU 虚拟化技术** → `gpu/gpu_manager/` 完整学习
2. **容器化与编排** → `gpu/gpu_manager/` 相关章节
3. **MLOps 实践** → `mlops/README.md` - 构建 MLOps 平台
4. **性能优化实践** → 各目录的实践练习和代码实现
5. **多 GPU 训练优化** → `pytorch/pytorch-1h.ipynb` 多 GPU 章节
6. **Triton GPU 编程** → `triton/README.md` - 高级 GPU 内核开发

### LLM 应用路径
1. **提示工程** → `prompt/README.md` - 学习提示设计和优化
2. **上下文工程** → `context-engineering/README.md` - 学习上下文管理
3. **RAG 技术** → `rag/README.md` - 构建检索增强生成系统
4. **AI 智能体** → `agent/README.md` - 开发智能体应用
5. **Qwen 模型实践** → `qwen/README.md` - 使用和微调大语言模型
6. **强化学习** → `reinforcement-learning/README.md` - 学习 RL 在 LLM 中的应用

### 可解释 AI 路径
1. **可解释 AI 基础** → `xai/README.md` - 了解可解释 AI 概念和方法
2. **白盒模型** → `xai/interpretable-ai-book/Chapter_02/` - 学习线性回归、决策树、GAM
3. **全局可解释性** → `xai/interpretable-ai-book/Chapter_03/` - 学习 PDP、特征交互
4. **局部可解释性** → `xai/interpretable-ai-book/Chapter_04/` - 学习 LIME、SHAP、Anchors
5. **视觉解释** → `xai/interpretable-ai-book/Chapter_05/` - 学习显著性映射、Grad-CAM
6. **公平性分析** → `xai/interpretable-ai-book/Chapter_08/` - 学习公平性分析和偏见缓解

## 🔗 相关资源

### 官方文档
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Kubernetes Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [Triton 官方文档](https://triton-lang.org/)
- [Qwen 官方文档](https://qwen.readthedocs.io/)

### 学习资源
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [PyTorch 示例](https://github.com/pytorch/examples)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Qwen 官方文档](https://qwen.readthedocs.io/)
- [DashScope API 文档](https://help.aliyun.com/zh/model-studio/)

### 社区支持
- [NVIDIA 开发者论坛](https://forums.developer.nvidia.com/)
- [PyTorch 论坛](https://discuss.pytorch.org/)
- [Stack Overflow - CUDA](https://stackoverflow.com/questions/tagged/cuda)
- [Stack Overflow - PyTorch](https://stackoverflow.com/questions/tagged/pytorch)
- [Hugging Face Forums](https://discuss.huggingface.co/)

## 📄 许可证

本项目采用 [LICENSE](LICENSE) 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来完善本项目。

---

**注意**：本项目为学习资料库，内容持续更新中。如有问题或建议，欢迎反馈。
