<mark> 附录 E - 命令行上的智能体（AI Agents on the CLI）</mark>

引言
--

开发者的命令行，长期以来是精确、命令式操作的堡垒，如今正经历深刻变革。它正在从简单的 Shell 演变为智能、协作的工作空间，由新一代工具——智能体命令行界面（CLI）驱动。这些智能体不再只是执行命令，而是能够理解自然语言、维护整个代码库的上下文，并执行复杂的多步任务，自动化开发生命周期中的大量环节。

本指南深入介绍了该领域的四个主流工具，分析它们的独特优势、理想场景和设计理念，帮助你选择最适合自身工作流的工具。需要注意的是，某个工具的用例往往也能被其他智能体实现，关键区别在于它们完成任务的质量、效率和细致程度。后文将介绍专门用于衡量这些能力的基准测试。

Claude CLI (Claude Code)
------------------------

Anthropic 的 Claude CLI 是为高阶编码场景打造的智能体，具备对项目架构的深度、整体理解。其核心优势在于“智能体”特性，能为你的仓库建立心智模型，处理复杂的多步任务。交互方式高度对话化，类似结对编程，智能体会在执行前解释计划，适合需要大规模重构或架构性变更的专业开发者。

**典型用例**：

1.   **大规模重构**：你可以指令：“当前用户认证依赖 session cookie，请将整个代码库重构为无状态 JWT，更新登录/登出接口、中间件和前端 token 处理。”Claude 会读取所有相关文件并协调修改。
2.   **API 集成**：提供新的天气服务 OpenAPI 规范后，可指令：“集成该天气 API，创建服务模块处理 API 调用，新增组件展示天气，并更新主仪表盘。”
3.   **文档生成**：指定复杂模块的代码，如：“分析 ./src/utils/data_processing.js，为每个函数生成详细 TSDoc 注释，包括用途、参数和返回值。”

Claude CLI 是专业编码助手，内置文件读取、代码结构分析和编辑生成等核心开发工具，并深度集成 Git，支持分支和提交管理。其可扩展性通过 Multi-tool Control Protocol（MCP）实现，用户可自定义工具，支持私有 API、数据库查询和项目脚本执行。开发者决定智能体的功能边界，Claude 本质上是推理引擎加用户定义工具的组合。

Gemini CLI
----------

Google 的 Gemini CLI 是一款功能强大且易用的开源智能体，依托先进的 Gemini 2.5 Pro 模型，拥有超大上下文窗口和多模态能力（支持图片与文本）。其开源特性、慷慨的免费额度和“推理 - 行动”循环，使其透明、可控，适合从爱好者到企业开发者，尤其是 Google Cloud 生态用户。

**典型用例**：

1.   **多模态开发**：你提供设计稿截图（gemini describe component.png），指令：“写出与此完全一致的 React 组件 HTML 和 CSS，确保响应式。”
2.   **云资源管理**：利用内置 Google Cloud 集成，指令：“查找生产项目中所有 GKE 集群，筛选版本低于 1.28 的，并生成逐个升级的 gcloud 命令。”
3.   **企业工具集成（MCP）**：开发者提供 get-employee-details 工具连接公司 HR API，指令：“为新员工撰写欢迎文档。先用 `get-employee-details --id=E90210` 获取姓名和团队，再填充 `welcome_template.md`。”
4.   **大规模重构**：需要将大型 Java 代码库的日志库替换为新框架，可指令：读取 ‘src/main/java’ 下所有 `*.java` 文件，将 ‘`org.apache.log4j`’ 及其 Logger 类替换为 ‘`org.slf4j.Logger`’ 和 LoggerFactory，重写日志实例化及 `.info()`、`.debug()`、`.error()` 调用为结构化格式。

Gemini CLI 内置多种工具，支持文件系统操作（读写）、Shell 命令执行、网络访问（抓取与搜索）、多文件读取和会话记忆。其安全性通过沙箱隔离模型行为，MCP 服务器则安全地连接本地环境或其他 API。

Aider
-----

Aider 是开源 AI 编码助手，像真正的结对程序员一样直接操作你的文件并提交 Git 变更。其最大特点是直接性：自动应用编辑、运行测试验证，并将每次成功修改自动提交。Aider 支持多种模型，用户可自由选择成本与能力。其以 Git 为核心的工作流，适合追求高效、可控和可审计代码变更的开发者。

**典型用例**：

1.   **测试驱动开发（TDD）**：开发者可指令：“为阶乘函数创建一个失败的测试。”Aider 编写测试并运行失败，下一步：“现在编写代码让测试通过。”Aider 实现函数并再次运行测试确认。
2.   **精准修复 Bug**：针对 bug 报告，指令：“`billing.py` 的 `calculate_total` 函数在闰年出错。添加文件到上下文，修复 bug，并用现有测试集验证。”
3.   **依赖更新**：指令：“项目使用过时的 ‘requests’ 库，请遍历所有 Python 文件，更新 import 和弃用函数调用，兼容最新版，并更新 `requirements.txt`。”

GitHub Copilot CLI
------------------

GitHub Copilot CLI 将流行的 AI 结对编程助手扩展到终端，最大优势是与 GitHub 生态的深度原生集成。它能理解项目在 GitHub 上的上下文，具备智能体能力，可被分配 Issue，自动修复并提交 Pull Request 供人工审核。

**典型用例**：

1.   **自动化 Issue 解决**：管理员分配 bug 工单（如“Issue #123：修复分页 off-by-one 错误”）给 Copilot Agent，智能体自动新建分支、编写代码并提交 PR，引用 Issue，无需人工干预。
2.   **仓库上下文问答**：新成员可问：“仓库中数据库连接逻辑在哪，需哪些环境变量？”Copilot CLI 利用仓库全局上下文，精确回答并给出文件路径。
3.   **Shell 命令助手**：用户不确定复杂命令时可问：gh? 找出所有大于 50MB 的文件，压缩并放入 archive 文件夹。Copilot 会生成所需 Shell 命令。

Terminal-Bench：命令行 智能体基准测试
--------------------------

Terminal-Bench 是专为评估智能体在命令行执行复杂任务能力而设计的新型框架。命令行因其文本化、沙箱特性，被认为是智能体的理想运行环境。首发版 Terminal-Bench-Core-v0 包含 80 个手工策划任务，涵盖科学工作流和数据分析等领域。为公平对比，开发了极简智能体 Terminus，作为各类语言模型的标准测试平台。框架支持容器化或直连集成多种智能体，未来将支持大规模并行评测和主流基准接入。项目鼓励开源贡献，扩展任务库和协作完善框架。

总结
--

这些强大的 AI 命令行智能体的出现，标志着软件开发方式的根本转变，终端正成为动态、协作的新环境。正如所见，没有绝对“最佳”工具，而是形成了各具专长的生态：Claude 适合复杂架构任务，Gemini 擅长多模态与通用问题，Aider 适合 Git 驱动的直接代码编辑，GitHub Copilot 则无缝融入 GitHub 工作流。随着工具不断进化，熟练运用它们将成为开发者必备技能，深刻改变软件的构建、调试和管理方式。

参考资料
----

*   [Anthropic – Claude CLI 文档](https://docs.anthropic.com/en/docs/claude-code/cli-reference)
*   [Google Gemini CLI – GitHub 仓库](https://github.com/google-gemini/gemini-cli)
*   [Aider – 官方网站](https://aider.chat/)
*   [GitHub Copilot CLI – 文档](https://docs.github.com/en/copilot/github-copilot-enterprise/copilot-cli)
*   [Terminal Bench – 项目主页](https://www.tbench.ai/)