[**中文**](README.md) | [English](README.en.md)

# Strata L4 · GSQ-RCO IQ3_XXS

> 权重来源：[alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF)

本项目在 Google Colab L4 High-RAM 环境运行 Qwen3.8-Flash-Next 的 GSQ-RCO IQ3_XXS 权重与拒绝方向控制向量，提供带鉴权的 OpenAI 兼容 Chat Completions API。项目用于个人交互推理、代码 agent 与长上下文实验，消耗已有 Google Colab Pro 订阅的每月计算额度。

入口文件为 [strata_l4_gsq_rco_iq3_xxs.py](strata_l4_gsq_rco_iq3_xxs.py)，已生成的 notebook 为 [strata_l4_gsq_rco_iq3_xxs.ipynb](strata_l4_gsq_rco_iq3_xxs.ipynb)。源文件使用 `# %%` 分隔单元格。

## 性能与硬件规范

在 L4 实例上，默认配置（关闭 MTP）实测对话解码速度约为 32 到 57 tokens/s（tps），峰值可达 80+ tps。实际速度受 CPU 性能、上下文长度、专家缓存命中率与草稿接受率影响。

| 项目 | 配置与指标 |
|---|---|
| 推荐实例 | L4 GPU + High-RAM |
| 运行时资源 | 约 22 GiB 可用显存，53 GiB 系统内存 |
| 默认量化 | IQ3_XXS，2 个分片文件共约 75.8 GB |
| 专家内存 | 专家权重约占 43 GB，启动前检查可用内存至少 48 GiB |
| 上下文长度 | 131,072 tokens，int8 KV streaming，显存缓存窗口 32,768 |
| KV 内存占用 | 完整 KV 缓存占用 1.55 GiB 系统内存 |
| 长输入测试 | 34,043 tokens，正确回忆开头标记，耗时约 60 秒 |
| 测试后剩余内存 | 约 6.4 GiB 可用系统内存 |

![运行时资源监控](screenshot/monitor.png)

项目未对完整 128K 输入进行压力测试。34K 测试耗时包含预填充耗时，不代表纯解码速度。服务默认仅处理文本输入。首次源码编译需要 10 到 30 分钟。磁盘检查要求剩余空间大于下载文件总大小加 24 GiB 工作空间。

## 模型权重与控制向量

模型权重来自 [alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF)，基于 [ISTA-DASLab GSQ-RCO 量化版本](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)，并附带从 Huihui 检查点提取的拒绝方向控制向量。权重文件未做修改。[Strata](https://github.com/Niko1221/Strata) 引擎在推理时动态投影控制向量，默认作用于第 4 到 44 层，强度设为 1.0。

原仓库作者测试了 IQ3_S 与 IQ2_XS 量化，未测试 IQ3_XXS。作者报告两档量化通过 5 个小型代码工具任务与 32K/115K 标记检索测试。在 RTX 3090 与 EPYC 平台上，32K 上下文的解码速度约为 59 与 87 tps。该测试结果仅供能力参考，不代表 Colab L4 的表现。详细数据参见[原仓库测量说明](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF#measurements)。

代码同时支持 IQ2_XS（约 68 GB 下载量，35.5 GB 专家内存）与 IQ3_S（约 83.6 GB 下载量，50 GB 专家内存）。在 53 GiB 内存的运行时中默认使用 IQ3_XXS。若内存紧张，可切换为 IQ2_XS。模型许可证参见 [Hugging Face LICENSE](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF/blob/main/LICENSE)。

## Colab Pro 额度计算

建议使用已有的 Colab Pro 或 Google AI Pro 订阅额度。启动 L4 High-RAM 运行时执行任务，使用完毕后立即断开并删除运行时。Google AI Pro 与 Colab Pro 为独立订阅，具体权益以 Google 账户控制台为准。

L4 High-RAM 运行时的历史消耗记录为 1.57 计算单元/小时。该数值为消耗速率，并非固定价格。

| 可用额度 | 理论 GPU 运行时间 |
|---:|---:|
| 100 计算单元 | 约 63.7 小时 |
| 200 计算单元 | 约 127.4 小时 |

源码编译、文件下载与空闲挂机均会扣除计算单元。上述时长为累计时间，单次会话可能因配额或资源回收提前终止。Colab Pro 官方月度基准为 100 单元。运行前请在资源面板确认实时费率与剩余额度。相关说明参见 [Google 产品文档](https://developers.google.com/workspace/admin/reseller/v1/how-tos/products#google_colab) 与 [Colab FAQ](https://research.google.com/colaboratory/faq.html)。

## 保持运行时活跃

模型在后台运行。若需防止 notebook 单元执行结束导致空闲断连，可在新单元运行以下健康检查循环。该循环每 5 分钟请求一次健康检查端点：

```python
import time

while True:
    try:
        status = request("/health")
        print(time.strftime("%H:%M:%S"), status, flush=True)
    except Exception as exc:
        print(time.strftime("%H:%M:%S"), type(exc).__name__, flush=True)
    time.sleep(300)
```

若需停止保活循环，点击 Colab 单元格的停止按钮。若需终止推理服务，运行 `stop_owned("tunnel")` 与 `stop_owned("server")`。Colab Pro+ 支持最长 24 小时连续运行，Colab Pro 仍受限于运行时策略与资源回收。保活循环无法阻止系统主动回收实例。详情参见 [Colab 官方运行时说明](https://research.google.com/colaboratory/faq.html)。请勿使用浏览器模拟点击脚本。

## 部署方法一：Colab MCP 与 Agent Skill

1. 安装 `uv` 工具：`pip install uv`。
2. 配置本地 agent 的 MCP 服务，参考 [googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp)。若客户端支持 `mcp.json`，添加以下配置：

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
```

3. 在浏览器登录 Google 账号，并通过 agent 打开 MCP 授权页面。客户端必须支持 `notifications/tools/list_changed`，以获取 notebook 操作工具。
4. 发送以下提示词给 agent：

```text
请读取本项目的 skills/strata-colab/SKILL.md，通过 Colab MCP 导入
strata_l4_gsq_rco_iq3_xxs.ipynb。选择 L4 High-RAM 运行时并按顺序执行，
验证推理与鉴权功能，最后输出客户端连接配置。使用默认的 IQ3_XXS 权重与 128K 上下文。
```

5. 若需 agent 自动调用该流程，将 `skills/strata-colab` 目录复制到 agent 的 skills 目录。部署完成后，记录生成的公网地址与 API 密钥。

若默认浏览器打开授权页失败，且本地 Chromium 已开启 9222 调试端口，运行 `python scripts/open_colab_browser.py '<URL>'`。连接 URL 包含私有凭证，请勿公开。

## 部署方法二：手动导入 Colab

1. 登录 [Colab](https://colab.research.google.com/)，选择“文件 → 上传笔记本”，上传 `strata_l4_gsq_rco_iq3_xxs.ipynb`。
2. 打开“运行时 → 更改运行时类型”，硬件加速器选择 L4，运行时配置选择 High-RAM，点击保存并连接。
3. 顺序执行全部 5 个代码单元：环境构建、权重下载、配置生成、启动与本地验证、公网隧道建立。
4. 复制末尾输出的 Base URL 与 Model 名称。可在 Colab Secrets 中预先配置 `STRATA_API_KEY`，并授权 notebook 读取。
5. 在本地客户端配置 Base URL、API 密钥与模型名称。若未预设密钥，运行最后一个单元查看自动生成的密钥。使用完毕后手动断开并删除运行时，避免额度消耗。

下载公开 Hugging Face 仓库无需 token。若需访问受限资源，可在 Colab Secrets 中添加 `HF_TOKEN`。API 密钥必须包含至少 24 位无空格 ASCII 字符。未配置密钥时系统会自动生成并存入私有文件。请勿在公开 notebook 中打印密钥。

## 配置参数与接口调用

修改第一个代码单元的变量后重新执行。修改上下文长度时，无需重建环境，仅需在当前内核更新变量，然后重新运行“配置生成”与“启动与本地验证”单元。启动单元会自动终止旧进程。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QUANT` | `IQ3_XXS` | 量化等级，可选 `IQ2_XS` 或 `IQ3_S` |
| `MAX_CONTEXT` | `131072` | 上下文总长度（包含输入与输出） |
| `KV_RESIDENT` | `32768` | 显存驻留 KV 窗口大小，其余放入系统内存 |
| `PREFILL` | `512` | 预填充批处理大小 |
| `ENABLE_MTP` | `False` | 是否启用 MTP，默认使用 lookup draft `--spec 2` |
| `PORT` | `8095` | 服务监听端口，避免冲突推荐使用 8095 到 8105 |

![Strata 引擎运行状态](screenshot/engine.png)

Strata 提供兼容 OpenAI Chat Completions（`POST /v1/chat/completions`）、Anthropic Messages 与 OpenAI Responses 的接口。详细说明参见 [Strata API 文档](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#using-it)。客户端使用 `Authorization: Bearer <key>` 请求头鉴权。可在终端通过 curl 测试：

```bash
export STRATA_BASE_URL='https://your-tunnel.trycloudflare.com/v1'
export STRATA_MODEL='alesha-qwen3.8-flash-next-gsq-rco-iq3_xxs'
read -rsp 'API key: ' STRATA_API_KEY
export STRATA_API_KEY
curl "$STRATA_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $STRATA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"alesha-qwen3.8-flash-next-gsq-rco-iq3_xxs","messages":[{"role":"user","content":"What is 17 + 25?"}],"max_tokens":128,"reasoning_effort":"none"}'
```

临时公网隧道的域名在重启后会变化。长输入的首个 token 预填充耗时较长，客户端需要配置足够长的请求超时。

若使用 Pi，将以下内容合并至 `~/.pi/agent/models.json` 的 `providers` 字段：

```json
{
  "strata-colab": {
    "baseUrl": "https://your-tunnel.trycloudflare.com/v1",
    "api": "openai-completions",
    "apiKey": "$STRATA_API_KEY",
    "models": [{
      "id": "alesha-qwen3.8-flash-next-gsq-rco-iq3_xxs",
      "name": "Strata Qwen3.8 Flash Next IQ3_XXS",
      "reasoning": true,
      "input": ["text"],
      "contextWindow": 131072,
      "maxTokens": 4096
    }]
  }
}
```

配置完成后，在终端导入密钥并启动 Pi：

```bash
read -rsp 'API key: ' STRATA_API_KEY; printf '\n'
export STRATA_API_KEY
pi
```

在 Pi 命令行运行 `/model`，选择 `strata-colab/alesha-qwen3.8-flash-next-gsq-rco-iq3_xxs`。`api` 设置为 `openai-completions`，`baseUrl` 必须以 `/v1` 结尾。`contextWindow` 对应服务的 131072 上限，`maxTokens` 设置单次输出最大 token 数。

可在 `settings.json` 中配置 `compaction.modelOverrides`，设置 `reserveTokens: 8192` 与 `keepRecentTokens: 32768`。当上下文接近 120K 时执行历史压缩。Pi 1.1.0 不支持独立压缩模型配置，若需单独摘要模型，需要扩展 `session_before_compact` 钩子。请在外部文件保存关键约束，防止压缩丢失细节。

## 项目维护

```text
strata_l4_gsq_rco_iq3_xxs.py       notebook 源码文件
strata_l4_gsq_rco_iq3_xxs.ipynb    构建生成的 notebook 文件，不含输出数据
scripts/build_notebook.py          notebook 构建脚本，仅依赖 Python 标准库
scripts/open_colab_browser.py      Chromium CDP 浏览器连接工具
tests/                             单元测试套件
skills/strata-colab/SKILL.md       Agent 自动化部署与排错流程
README.md / README.en.md           中文与英文文档
```

修改 `.py` 源码后，运行以下命令同步并验证：

```bash
python scripts/build_notebook.py
python -m unittest discover -s tests -p 'test_*.py' -q
```

Strata 引擎与 llama.cpp 的版本已在源码中锁定至特定提交。下载阶段会验证分片大小与 GGUF 文件头，不做完整文件 SHA256 校验。编译输出、配置文件与进程日志存放于 `/content/Strata-data`，引擎代码存放于 `/content/Strata-colab`。Colab 实例重置后会清空上述临时文件，系统不会将数据同步至 Google Drive。
