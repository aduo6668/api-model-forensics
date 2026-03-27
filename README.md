# API Model Forensics

Evidence-driven, local-first model identity audit for AI relays, gateways, and model websites.

`API Model Forensics` 是一个本地优先的模型法证项目。它的目标不是假装“100% 猜出底模”，而是把“这个 API 到底是不是它声称的那个模型”这件事，尽可能做成一个可复核、可复现、讲证据层级的审计流程。

我们希望它能帮助用户识别几类真实风险：

- 挂羊头卖狗肉：页面写的是高质量模型，实际接的是别的模型家族。
- 静默降级：高峰期、低价套餐或风控条件下，后台悄悄切到更弱模型。
- 混模与动态路由：同一个模型名背后，实际上按时段、任务或成本在切多个后端。
- 包装层误导：底模可能没变，但 system prompt、代理层或安全层把行为改得像另一个模型。

## Vision

这个项目的愿景，是把“模型识别”从截图、传言和主观感觉，推进成一种更接近技术审计的工作方式。

我们会持续整理论文、官方材料、公开案例和现实信号，把其中真正能落地的检测手段沉淀为本地工具。最终目标不是做一个神秘的“底模算命器”，而是做一个面对 API relay、AI gateway 和模型站点的证据型审计产品：给出概率、证据分层、主要 caveat 和可复核报告，而不是只给一句武断结论。

## What The MVP Does

当前 MVP 已经可以：

- 启动本地桌面窗口，输入 `API 地址 / API Key / 模型名` 即可运行检测。
- 对 OpenAI-compatible chat/completions 端点发起一组低成本 probes。
- 输出四类核心概率：
  - 与申报模型一致
  - 同家族降级
  - 其他模型家族
  - 包装 / 混模 / 未知
- 给出结论标签、证据分层和主要 caveat。
- 自动生成本地 PDF 报告和 JSON 结果文件。

## Method At A Glance

MVP 不是靠单个 prompt 拍脑袋判断，而是把证据拆成几层一起看：

- `Protocol surface`
  - 看 `/models`、错误结构、tool schema、finish reason、参数兼容性等接口表面指纹。
- `Tokenizer / token accounting`
  - 看回显、usage 统计、截断行为和分词体系线索是否自洽。
- `Behavior probes`
  - 看结构化输出、精确格式、中文/英文/代码混合任务、轻度 policy 边界行为。
- `Stability / routing`
  - 看重复采样是否明显漂移，是否存在混模、fallback 或动态路由痕迹。

这个方向来自我们对一批论文与公开材料的整理，MVP 目前采用的是低成本、工程可落地的版本，而不是一上来就做重型研究管线。更细的研究和 probe 设计见：

公共仓库默认只保留产品代码、运行入口和用户文档。本地研究语料、论文下载、内部设计稿和实验备忘默认留在用户本机，不随仓库公开发布。

## Privacy And Local-Only Promise

这个项目以“本地优先、用户自控”为默认原则：

- 本工具没有自己的云端后端，也不会把检测数据上传到“我们的服务器”。
- 运行日志、JSON 结果和 PDF 报告默认只保存在用户本地。
- API Key 由用户自己提供、自己控制。
- 网络请求只会发送到你在界面里配置的目标 API 地址。
- 结果是概率型法证结论，不承诺 `100%` 还原真实底模。

如果你把地址填成远程 API，那么你的请求当然仍会发送到那个目标服务商；本项目的承诺是：**不会额外把数据转存、转传或上报到别的云端。**

## Cost Guardrails

我们把检测成本视为产品设计的一部分，而不是事后补充说明。

- `fast`
  - 预计约 `12,000` tokens
  - 建议上限 `15,000`
- `standard`
  - 预计约 `28,000` tokens
  - 建议上限 `40,000`
- `deep`
  - 预计约 `52,000` tokens
  - 建议上限 `80,000`

默认设计目标是始终远低于 `1,000,000` tokens。对于多数场景，我们优先使用便宜 probe 先收集强信号，只有在必要时才加深采样。

## Quick Start

### 1. Install

```powershell
cd C:\AI\API_Oasis\vendors\api-model-forensics
pip install -r requirements.txt
```

### 2. Launch

双击启动：

- [launcher/start.bat](launcher/start.bat)

如果你想看控制台输出，用：

- [launcher/start-dev.bat](launcher/start-dev.bat)

也可以直接运行：

```powershell
python -m app.main
```

### 3. Enter Inputs

在窗口里填入：

- `API 地址`
- `API Key`
- `模型名`
- 可选的 `provider hint`
- 检测档位：`fast / standard / deep`

### 4. Run And Review

运行完成后，你会看到：

- 结论标签
- 四类概率
- Top candidates
- 证据分层
- 主要 caveat
- 本地 PDF 报告路径

## Optional Local Defaults

如果你经常测试同一个目标，可以在项目根目录创建本地 `.env`，让 GUI 自动预填。这个文件已经被 Git 忽略，不会默认上传到仓库。

```env
API_FORENSICS_BASE_URL=http://127.0.0.1:8317
API_FORENSICS_API_KEY=your-key
API_FORENSICS_MODEL=gpt-5.4
API_FORENSICS_PROVIDER_HINT=
API_FORENSICS_MODE=standard
```

## Output Artifacts

每次运行会在本地生成一组可复核产物：

- `outputs/runs/<timestamp-model>/request_log.json`
- `outputs/runs/<timestamp-model>/summary.json`
- `outputs/runs/<timestamp-model>/normalized_outputs.json`
- `outputs/runs/<timestamp-model>/report.json`
- `outputs/reports/<timestamp-model>-report.pdf`

这些文件可以帮助你复盘某次检测为什么得到那个结论，也方便后续做横向对比。

## How To Read The Result

建议这样理解结果，而不是把它当成“绝对判决”：

- `likely consistent with claimed model`
  - 当前证据整体支持“与申报模型基本一致”。
- `likely same-family downgrade`
  - 更像同家族但更弱、成本更低或更小的模型。
- `likely alternative family`
  - 证据显示它更像另一个模型家族。
- `likely wrapped or policy-overlaid`
  - 可能存在明显包装层，行为被 system prompt 或策略层扭曲。
- `suspected routing shift or mixed backend`
  - 重复采样出现漂移，疑似混模、fallback 或动态路由。
- `ambiguous`
  - 现有证据不足以支持高置信判断。

## Current Scope

当前版本聚焦在一个清晰、可交付的本地 MVP：

- Windows 本地桌面启动
- OpenAI-compatible chat/completions 端点
- 低成本 probe suite
- 本地 JSON + PDF 报告
- 研究结论到工程实现的最小闭环

还没有追求的内容包括：

- 研究级 TRAP / RESF 全量实现
- 精确 snapshot 归因
- 重型网络侧流量指纹
- 云端托管与持续监控服务

## Repository Guide

- [app](app)
  - 桌面 UI、probe runner、scoring、report 生成逻辑
- [launcher](launcher)
  - Windows 启动脚本
- [requirements.txt](requirements.txt)
  - 运行依赖
- [.gitignore](.gitignore)
  - 本地输出、密钥与研究资料忽略规则

## Notes

- 当前 MVP 接受 `mock://openai`、`mock://anthropic`、`mock://gemini`、`mock://mixed` 作为本地烟测输入。
- 如果你准备把这个项目公开发布，建议优先检查 `.env`、运行产物和测试样本是否已被正确忽略。
- 研究语料、论文 PDF、文本提取和内部设计文档默认只保留在本地，不进入公共仓库。
- 这是一个追求“证据质量”和“用户成本”平衡的工程项目，不是营销页，也不是万能鉴模神谕。
