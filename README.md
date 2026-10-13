[**中文**](README.md) | [English](README.en.md)

# Strata L4 · GSQ-RCO IQ3_XXS

> 模型卡与权重来源：[alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF)

在 **Colab L4 High-RAM** 上运行 Qwen3.8-Flash-Next 的 GSQ-RCO IQ3_XXS 权重与拒绝方向控制向量，提供带鉴权的 OpenAI Chat Completions API。项目用于个人交互式推理、代码 agent 和长对话实验，让已有 Pro 订阅中的每月 Colab 额度得到实际使用。

入口是 [strata_l4_gsq_rco_iq3_xxs.py](strata_l4_gsq_rco_iq3_xxs.py)，可直接上传的 notebook 是 [strata_l4_gsq_rco_iq3_xxs.ipynb](strata_l4_gsq_rco_iq3_xxs.ipynb)。Python 文件使用 `# %%` 分隔单元，生成 notebook 后在 Colab 中执行。

## 性能与硬件

**预期解码目标：57 tokens/s（tps）。** 这是项目目标，不是已完成的 L4 基准结果。当前默认配置关闭 MTP，已有 L4 对话日志约为 32–42 tps。速度受 CPU、上下文、专家缓存命中率和草稿接受率影响。

| 项目 | 配置或实测结果 |
|---|---|
| 推荐实例 | L4 GPU + High-RAM |
| 本次运行时 | 约 22 GiB 可用显存，53 GiB 系统内存 |
| 默认量化 | IQ3_XXS，两个 GGUF 分片共约 75.8 GB |
| 系统内存 | 专家约 43 GB，启动前检查至少 48 GiB 可用内存 |
| 上下文 | 131,072 tokens，int8 KV streaming，显存缓存窗口 32,768 |
| KV 内存 | 引擎日志显示完整 KV 缓存占 1.55 GiB 系统内存 |
| 长输入实测 | 34,043 tokens，正确回忆开头标记，耗时约 60 秒 |
| 测试后余量 | 约 6.4 GiB 可用系统内存 |

完整 128K 输入尚未压测。34K 测试耗时包含预填充，不能作为解码 tps。默认仅提供文本服务。首次编译约需 10–30 分钟，下载时间取决于网络。磁盘检查按缺失下载文件总量加 24 GiB 工作空间执行。

## 模型仓库与能力概览

权重来自 [alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF)。它复用 [ISTA-DASLab 的 GSQ-RCO 量化](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)，并附带从 Huihui checkpoint 提取的控制向量。权重本身保持原样；[Strata](https://github.com/Niko1221/Strata) 在运行时做投影，默认层范围 4–44、强度 1.0。

仓库作者测试了 IQ3_S 与 IQ2_XS，没有测试 IQ3_XXS。作者报告两种已测档位通过五个小型代码工具任务，以及 32K/115K 长文标记查找。32K 下解码约为 59/87 tps，测试机器是 RTX 3090 + EPYC，而非 Colab L4。这些结果提供能力参考，不能直接代表本项目默认量化的准确率或速度。[测量说明](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF#measurements)

本项目另支持 IQ2_XS（约 68 GB 下载、35.5 GB 专家 RAM）和 IQ3_S（约 83.6 GB、50 GB）。53 GiB RAM 的运行时优先使用 IQ3_XXS；如果内存不足，可选 IQ2_XS。模型许可见 [HF LICENSE](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF/blob/main/LICENSE)，引擎许可见其上游仓库。

## 使用 Pro 每月额度

推荐先消耗已有 **Colab Pro 或含 Colab 权益的 Google AI Pro** 每月额度，按需启动 L4 High-RAM，用完后在 Colab 中断开并删除运行时。Google AI Pro 与 Colab Pro 是不同订阅，账户的实际权益以 Colab 页面为准。

本项目使用的 L4 High-RAM 消耗记录为 **1.57 计算单元/小时**，不是美元价格，也不是固定费率。按此记录估算：

| 可用额度 | 理论 GPU 使用时间 |
|---:|---:|
| 100 计算单元 | 约 63.7 小时 |
| 200 计算单元 | 约 127.4 小时 |

编译、下载和空闲连接也会消耗额度，应计入预算。以上是累计使用时间，不代表单次会话可持续这么久。Google 官方列出的 Colab Pro 月额度为 100 单元；Google AI 计划的权益和硬件可用性可能变化。启动前查看资源面板的实际费率与余额。[官方计划说明](https://developers.google.com/workspace/admin/reseller/v1/how-tos/products#google_colab)、[Colab FAQ](https://research.google.com/colaboratory/faq.html)

## Colab 持续运行（自动保活）

模型服务启动后会在后台运行。要让 notebook runtime 在使用期间保持执行状态，可在新代码单元运行下面的健康检查循环。它每 5 分钟检查服务并输出状态，避免该单元执行结束；这会持续消耗 L4 计算单元。

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

回到此单元，按 Colab 的停止按钮中断循环。需要停止推理服务时，再运行 `stop_owned("tunnel")` 和 `stop_owned("server")`。Colab Pro+ 在额度充足时支持最长 24 小时连续执行；Colab Pro 的实际时长受运行时上限和资源回收影响，保活循环不保证实例不会被回收。查看 [Colab 官方运行时说明](https://research.google.com/colaboratory/faq.html)。避免用浏览器脚本模拟点击或绕过 Colab 空闲限制。

## 方法一：Colab MCP + agent skill

1. 安装 `uv`：`pip install uv`。
2. 按 [googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp) 配置本地 agent 的 MCP 服务。支持 `mcp.json` 的客户端可使用：

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

3. 在浏览器中登录 Google，并让 agent 打开 MCP 连接页面。客户端需要支持 `notifications/tools/list_changed`，连接后才能发现 notebook 工具。
4. 指挥 agent 读取本项目的 [skills/strata-colab/SKILL.md](skills/strata-colab/SKILL.md)：

   ```text
   请读取此项目的 skills/strata-colab/SKILL.md，通过 Colab MCP 导入
   strata_l4_gsq_rco_iq3_xxs.ipynb，选择 L4 High-RAM，按顺序执行，
   检查推理和鉴权，并给出客户端配置。使用默认 IQ3_XXS、128K 上下文。
   ```

5. 如需 agent 自动发现该 skill，将整个 `skills/strata-colab` 文件夹安装到该 agent 的 skills 目录。配置客户端时使用本次运行生成的地址与 key。

MCP 本身不要求 Chromium CDP。仅在默认浏览器打开失败、且本地 Chromium 已启用 CDP 9222 时，可用 `python scripts/open_colab_browser.py '<MCP 返回的连接 URL>'` 打开连接页面。连接 URL 含私有 token，请勿分享。

## 方法二：手动导入 Colab

1. 打开 [Colab](https://colab.research.google.com/)，选择「文件 → 上传笔记本」，上传本项目 `.ipynb`。
2. 选择「运行时 → 更改运行时类型」，设置 **L4** 和 **高 RAM / High-RAM**，然后连接。
3. 按顺序执行五个代码单元：环境与编译 → 下载 → pack 与配置 → 启动与本地测试 → 公网隧道与测试。
4. 复制最后输出的 Base URL 和 Model。推荐预先在 Colab Secrets 设置 `STRATA_API_KEY`，并允许 notebook 访问。
5. 在客户端填入 Base URL、API key、模型 ID。需要取 key 时运行 notebook 最后的单元。该输出含密钥，请勿分享或保存带有此输出的 notebook。使用完毕后删除运行时，停止额度消耗。

公共 HF 仓库默认无需 token。受限下载可设置 Colab Secret `HF_TOKEN`。API key 必须至少 24 位、不含空白的 ASCII 字符；未设置时自动生成并保存在运行时私有文件中。不要在共享 notebook 中打印 key。

## 配置与调用

在第一个代码单元修改变量后重新执行。运行中的服务会占用内存，因此仅修改上下文时，在当前内核更新变量，再重跑「pack 与配置」和「启动与本地测试」。启动单元会停止本项目记录的旧服务进程。

| 变量 | 默认值 | 用途 |
|---|---|---|
| `QUANT` | `IQ3_XXS` | 可选 `IQ2_XS`、`IQ3_S` |
| `MAX_CONTEXT` | `131072` | 输入与输出共用的上下文上限 |
| `KV_RESIDENT` | `32768` | KV 在显存中的驻留窗口，完整缓存放在 RAM |
| `PREFILL` | `512` | 预填充批次大小 |
| `ENABLE_MTP` | `False` | 默认 lookup draft `--spec 2`，启用 MTP 需额外下载与内存 |
| `PORT` | `8095` | 避开可能由 Colab node 占用的 8080，忙时选择 8095–8105 |

Strata 同时提供 OpenAI Chat Completions（`POST /v1/chat/completions`）、Anthropic Messages 和 OpenAI Responses 等接口。[Strata API 文档](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#using-it)列出了端点、请求示例与认证方式。本项目的 Pi 配置选择 `openai-completions`，因此使用 Chat Completions 路径。客户端通过 `Authorization: Bearer <key>` 鉴权。先在终端设置环境变量：

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

临时隧道重建后域名可能变化。Strata 面板的 key 设置位于 **About → Settings**。长输入的首次预填充可能较慢，客户端需要足够长的超时。

Pi 可在 `~/.pi/agent/models.json` 的 `providers` 中合并以下配置，保留已有 provider：

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

保存后，在启动 Pi 的同一终端安全输入 key 并启动 Pi：

```bash
read -rsp 'API key: ' STRATA_API_KEY; printf '\n'
export STRATA_API_KEY
pi
```

在 Pi 中运行 `/model`，选择 `strata-colab/alesha-qwen3.8-flash-next-gsq-rco-iq3_xxs`。`api` 字段是 Pi 的传输格式选择：`openai-completions` 表示向 Strata 的 `/v1/chat/completions` 发送 OpenAI Chat Completions 请求；`baseUrl` 要以 `/v1` 结尾。`contextWindow` 应与服务的 131072 上限一致，`maxTokens` 是每次回复的输出上限。Strata 的端点与请求格式见[官方 API 文档](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#using-it)。

可在 `settings.json` 合并 `compaction.modelOverrides`，为此模型设置 `reserveTokens: 8192`、`keepRecentTokens: 32768`，约 120K 时压缩历史。Pi 1.1.0 没有配置项可直接指定另一个压缩模型；若要用独立摘要模型，需要扩展 `session_before_compact`。摘要可能遗漏细节，关键约束建议另存文件。

## 项目维护

```text
strata_l4_gsq_rco_iq3_xxs.py       notebook 单一源文件
strata_l4_gsq_rco_iq3_xxs.ipynb    可上传的生成文件，不含运行输出
scripts/build_notebook.py          仅用 Python 标准库生成 notebook
scripts/open_colab_browser.py      可选 Chromium CDP 连接助手
tests/                            配置、下载、进程与连接测试
skills/strata-colab/SKILL.md        agent 部署与排错流程
README.md / README.en.md           中文 / 英文说明
```

修改 `.py` 后执行：

```bash
python scripts/build_notebook.py
python -m unittest discover -s tests -p 'test_*.py' -q
```

引擎与 llama.cpp 固定到源文件中的提交。模型下载检查分片大小和 GGUF 文件头，未执行完整文件 SHA256 校验。编译、pack 和进程记录位于 `/content/Strata-data`，引擎位于 `/content/Strata-colab`。Colab 回收运行时会删除这些临时数据，本项目不自动备份到 Drive。
