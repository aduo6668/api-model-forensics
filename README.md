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

## Research Inspiration

这个项目不是“凭感觉发明一套鉴模方法”，而是明确受既有研究与评测框架启发。出于开源协作和致谢的基本态度，我们在公共仓库里直接标出主要参考来源，而不是把它们含糊地写成“内部整理”。

当前 MVP 主要参考这些工作：

- [TRAP: Targeted Random Adversarial Prompt Honeypot for Black-Box Identification](https://arxiv.org/abs/2402.12991)
  - 给我们提供了黑盒身份验证与低交互鉴别的核心问题设定。
- [Hide and Seek: Fingerprinting Large Language Models with Evolutionary Learning](https://arxiv.org/abs/2408.02871)
  - 给我们提供了“如何构造更有区分力的 prompts”这条思路。
- [A Fingerprint for Large Language Models](https://arxiv.org/abs/2407.01235)
  - 给我们提供了模型指纹、相似空间与黑盒比对的启发。
- [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110)
  - 给我们提供了更透明、更可复核的评测组织方式、记录方式和报告思路。
- [Robust and Scalable Bayesian Online Changepoint Detection](https://arxiv.org/abs/2302.04759)
  - 给我们提供了对重复采样漂移、路由变化和时段切换做变化检测的参考。

当前公开仓库展示的是受这些论文启发后的工程化 MVP，而不是对原论文的逐篇复现。任何实现上的简化、取舍和误差，责任都在本项目，不在原作者。

公共仓库默认只保留产品代码、运行入口和用户文档；论文下载、本地文本提取、内部设计稿和实验备忘默认留在用户本机，不随仓库公开发布。

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

## CLI Usage

除了 GUI，这个项目现在也支持命令行检测，适合脚本、自动化流程和其他 AI 工具直接调用。

查看帮助：

```powershell
python -m app.cli --help
```

返回机器可读 JSON：

```powershell
python -m app.cli `
  --base-url http://127.0.0.1:8317 `
  --api-key your-key `
  --model gpt-5.4 `
  --mode standard `
  --format json
```

返回适合人看的文本摘要：

```powershell
python -m app.cli `
  --base-url http://127.0.0.1:8317 `
  --api-key your-key `
  --model gpt-5.4 `
  --mode standard `
  --format text
```

如果你想从批处理入口打开 CLI 帮助，也可以用：

- [launcher/start-cli.bat](launcher/start-cli.bat)

## Maintenance And Updates

This repository is maintained with a local-first update model. Public code stays small and stable, while provider adapters, aliases, baselines, and probe updates are added incrementally during explicit maintenance windows.

- `Maintenance policy`: [docs/MAINTENANCE_PLAN.md](docs/MAINTENANCE_PLAN.md)
- `Local-first`: raw requests, outputs, baselines, and internal notes stay on the user's machine by default
- `Token discipline`: new probes should stay cheap unless a deeper pass is justified by evidence
- `Onboarding rule`: add new vendors or model families with additive registry changes and regression checks, not broad rewrites

## External Catalog Refresh

The project can also pull a frontier catalog snapshot from OpenRouter for local research and candidate expansion. This is a supplemental source for fast-moving model names and descriptions, not a replacement for first-party provider docs.

- CLI: `python -m app.catalog_cli --source openrouter --limit 30`
- Batch launcher: [launcher/refresh-openrouter-catalog.bat](launcher/refresh-openrouter-catalog.bat)
- Local outputs: `outputs/catalogs/openrouter-catalog-latest.json`, `outputs/catalogs/openrouter-catalog-latest-summary.json`, `outputs/catalogs/openrouter-catalog-latest-summary.md`
- Expected use: use OpenRouter snapshots to discover new frontier aliases and descriptions, then decide what should graduate into the source-backed seed registry

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
