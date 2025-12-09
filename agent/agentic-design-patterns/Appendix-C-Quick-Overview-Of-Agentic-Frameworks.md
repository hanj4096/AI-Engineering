# 附录C: 智能体框架快速概览（Quick overview of Agentic Frameworks）</mark>


LangChain
---------

LangChain 是一个用于开发由大语言模型（LLM）驱动应用的框架。其核心优势在于 LangChain 表达式语言（LCEL），可以将各个组件“管道化”串联成链，实现清晰的线性流程：每一步的输出作为下一步的输入。适用于有向无环图（DAG）式的工作流，即流程单向且无循环。

典型应用场景：

*   简单 RAG：检索文档、生成提示词、获取 LLM 答案
*   摘要生成：输入用户文本，调用摘要提示词，返回结果
*   信息抽取：从文本中提取结构化数据（如 JSON）

```Python
# 一个简单的 LCEL 链式流程（示意代码，非可运行）
chain = prompt | model | output_parse
```

### LangGraph

LangGraph 是基于 LangChain 构建的高级智能体系统库。它允许将工作流定义为图结构，节点为函数或 LCEL 链，边为条件逻辑。最大优势是支持循环，可灵活实现任务重试、工具调用等，直到目标达成。LangGraph 显式管理应用状态，状态对象在节点间传递并不断更新。

典型应用场景：

*   多智能体系统：主管智能体分派任务给各专职智能体，可循环直到目标完成
*   计划 - 执行智能体：智能体 制定计划，执行步骤，根据结果循环更新计划
*   人类参与：流程可等待人工输入后决定下一步节点

| 功能 | LangChain | LangGraph |
| --- | --- | --- |
| 核心抽象 | 链（LCEL） | 节点图 |
| 工作流类型 | 线性（DAG） | 循环（支持环路） |
| 状态管理 | 每次运行基本无状态 | 显式持久状态对象 |
| 主要用途 | 简单、可预测流程 | 复杂、动态、状态化智能体 |

### 如何选择？

*   当应用流程清晰、可预测、线性时，选用 LangChain（LCEL）；如果流程从 A 到 B 到 C，无需回环，LangChain 是理想选择。
*   当需要智能体具备推理、规划、循环操作能力时，选用 LangGraph。比如智能体需调用工具、反思结果、尝试不同方案，则需 LangGraph 的循环与状态管理。

📄 LangGraph 并行流程示例

```Python

# 图状态
class State(TypedDict):
   topic: str
   joke: str
   story: str
   poem: str
   combined_output: str

# 节点
def call_llm_1(state: State):
   """首次 LLM 调用，生成笑话"""
   msg = llm.invoke(f"Write a joke about {state['topic']}")
   return {"joke": msg.content}

def call_llm_2(state: State):
   """第二次 LLM 调用，生成故事"""
   msg = llm.invoke(f"Write a story about {state['topic']}")
   return {"story": msg.content}

def call_llm_3(state: State):
   """第三次 LLM 调用，生成诗歌"""
   msg = llm.invoke(f"Write a poem about {state['topic']}")
   return {"poem": msg.content}

def aggregator(state: State):
   """将笑话和故事等合并输出"""
   combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
   combined += f"STORY:\n{state['story']}\n\n"
   combined += f"JOKE:\n{state['joke']}\n\n"
   combined += f"POEM:\n{state['poem']}"
   return {"combined_output": combined}

# 构建流程
parallel_builder = StateGraph(State)
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# 展示流程
display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

# 执行
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])
```

上述代码定义并运行了一个 LangGraph 并行流程，主要用于同时生成关于某主题的笑话、故事和诗歌，并合并输出。

Google 的 ADK
------------

Google 的 Agent Development Kit（ADK）是一个高层次、结构化的多智能体应用开发与部署框架。与 LangChain、LangGraph 不同，ADK 更偏向生产级智能体协作编排，而不是底层智能体逻辑构建。

LangChain 提供最基础的组件和标准接口，适合串联模型调用、结果解析等操作。LangGraph 则通过图结构实现更灵活的控制流，开发者可显式定义节点（函数/工具）和边（执行路径），支持复杂循环推理和状态管理，适合精细化智能体思维流程或多智能体系统。

Google ADK 则屏蔽了底层图结构，提供预设的多智能体架构模式。例如内置 `SequentialAgent`、`ParallelAgent` 等智能体类型，自动管理智能体间的控制流。ADK 以“团队”概念为核心，主智能体可分派任务给专职智能体，状态和会话管理更隐式，开发者无需手动传递状态对象。相比 LangGraph 的显式状态传递，ADK 更适合快速搭建协作型智能体工厂流水线。

```Python
from google.adk.agents import LlmAgent
from google.adk.tools import google_Search

dice_agent = LlmAgent(
   model="gemini-2.0-flash-exp", 
   name="question_answer_agent",
   description="A helpful assistant agent that can answer questions.",
   instruction="""Respond to the query using google search""",
   tools=[google_search],
)
```

此代码创建了一个具备搜索增强能力的智能体。收到问题后，Agent 会调用 Google Search 工具获取实时信息，并据此作答。

Crew.AI
---------------
CrewAI 提供了一个以协作角色和结构化流程为核心的多智能体编排框架。它抽象层次更高，开发者无需定义底层图结构，而是定义团队成员及分工，框架自动管理智能体间的交互。

核心组件包括智能体、Task 和 Crew。Agent 不仅有功能，还具备角色、目标和背景故事，影响其行为和沟通风格。Task 是分配给智能体的具体工作单元，包含描述和期望输出。Crew 是团队容器，包含所有智能体和任务，并执行预设流程（如顺序或分层流程）。顺序流程下，任务按顺序传递；分层流程下，类似经理智能体协调分派任务。

与其他框架相比，CrewAI 不再关注 LangGraph 那种显式状态管理和流程控制，而是让开发者设计团队章程。Google ADK 提供全生命周期平台，CrewAI 则专注于智能体协作逻辑和团队模拟。

```Python
@crew
def crew(self) -> Crew:
   """创建研究团队"""
   return Crew(
     agents=self.agents,
     tasks=self.tasks,
     process=Process.sequential,
     verbose=True,
   )
```

此代码为一组智能体设置了顺序流程，按任务列表依次执行，并开启详细日志监控进度。

其他智能体开发框架
---------

*   **Microsoft AutoGen**：AutoGen 以多智能体协作对话为核心，支持不同能力智能体间的复杂任务分解与协作。优势是灵活的对话驱动，可实现动态多智能体交互，但执行路径不确定性较高，对 prompt 工程要求较高。

*   **LlamaIndex**：LlamaIndex 是数据框架，专注于将 LLM 与外部/私有数据源连接，擅长构建数据摄取与检索管道，适合构建具备 RAG 能力的智能体。其数据索引与查询能力强，但复杂智能体控制流和多智能体编排能力不如专用智能体框架，适合以数据检索为核心的场景。

*   **Haystack**：Haystack 是开源框架，专注于大规模、生产级搜索系统。架构由可组合节点组成，支持文档检索、问答、摘要等管道。优势是性能和可扩展性，适合企业级信息检索，但对于高度动态和创造性智能体行为实现较为刚性。

*   **MetaGPT**：MetaGPT 通过预设 SOP（标准操作流程）分配智能体角色和任务，模拟软件公司团队协作。优势是结构化输出，适合代码生成等专业领域，但高度专用，通用智能体任务适应性较弱。

*   **SuperAGI**：SuperAGI 是开源框架，提供智能体全生命周期管理，包括智能体配置、监控和可视化界面，提升智能体执行可靠性。优势是生产级特性和故障处理，但平台

*   **Microsoft AutoGen**：AutoGen 以多智能体协作对话为核心，支持不同能力智能体间的复杂任务分解与协作。优势是灵活的对话驱动，可实现动态多智能体交互，但执行路径不确定性较高，对 prompt 工程要求较高。

*   **LlamaIndex**：LlamaIndex 是数据框架，专注于将大语言模型与外部/私有数据源连接，擅长构建数据摄取与检索管道，适合构建具备 RAG 能力的智能体。其数据索引与查询能力强，但复杂智能体控制流和多智能体编排能力不如专用智能体框架，适合以数据检索为核心的场景。

*   **Haystack**：Haystack 是开源框架，专注于大规模、生产级搜索系统。架构由可组合节点组成，支持文档检索、问答、摘要等管道。优势是性能和可扩展性，适合企业级信息检索，但对于高度动态和创造性智能体行为实现较为刚性。

*   **MetaGPT**：MetaGPT 通过预设 SOP（标准操作流程）分配智能体角色和任务，模拟软件公司团队协作。优势是结构化输出，适合代码生成等专业领域，但高度专用，通用智能体任务适应性较弱。

*   **SuperAGI**：SuperAGI 是开源框架，提供智能体全生命周期管理，包括智能体配置、监控和可视化界面，提升智能体执行可靠性。优势是生产级特性和故障处理，但平台较为复杂，可能带来更多运维和集成成本。

*   **Semantic Kernel**：由微软开发，Semantic Kernel 是一个 SDK，通过插件和规划器系统将大语言模型与传统编程代码集成。它允许 LLM 调用原生函数并编排工作流，适合将模型作为推理引擎嵌入企业应用。优势是与 .NET 和 Python 代码库的无缝集成，但插件和规划器架构学习曲线较高。

*   **Strands Agents**：AWS 推出的轻量级 SDK，采用模型驱动方式构建和运行智能体，支持从基础对话助手到复杂多智能体系统。框架对模型供应商兼容性好，并原生集成 MCP 工具调用。优势是简单灵活，易于定制智能体循环，但由于设计轻量，开发者需自行完善监控和生命周期管理等运维能力。

总结
--

智能体框架生态极为丰富，从底层智能体逻辑库到高层多智能体协作平台应有尽有。LangChain 适合线性流程，LangGraph 支持复杂循环推理和状态管理。CrewAI、Google ADK 等高层框架聚焦团队协作和角色分工，LlamaIndex 则专注数据密集型场景。开发者需根据项目需求权衡底层控制力与平台易用性，选择适合的抽象层级。随着生态不断演进，开发者可灵活选用合适工具，构建更复杂的智能体系统。

参考资料
----

*   [LangChain - langchain.com](https://www.langchain.com/)
*   [LangGraph - langchain.com/langgraph](https://www.langchain.com/langgraph)
*   [Google ADK - google.github.io/adk-docs](https://google.github.io/adk-docs/)
*   [CrewAI - docs.crewai.com](https://docs.crewai.com/en/introduction)


