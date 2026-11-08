[中文](README.md) | [**English**](README.en.md)

# Strata L4 · GSQ-RCO IQ3_XXS

> Model card and weight source: [alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF)

Run Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS with a refusal-direction control vector on **Colab L4 High-RAM**. Get an authenticated OpenAI Chat Completions API for personal interactive inference, coding agents, and long conversations. Use the monthly Colab allowance in an existing Pro subscription.

[strata_l4_gsq_rco_iq3_xxs.py](strata_l4_gsq_rco_iq3_xxs.py) is the source, with `# %%` cell boundaries. Upload the generated [strata_l4_gsq_rco_iq3_xxs.ipynb](strata_l4_gsq_rco_iq3_xxs.ipynb) to Colab.

## Performance and hardware

On L4 instances with the previous 128K configuration and MTP disabled, measured conversation decode speeds range from 32 to 57 tokens/s (tps), with peak speeds reaching 80+ tps. CPU performance, context length, expert cache hit rate, and draft acceptance rate affect speed.

| Item | Configuration or measurement |
|---|---|
| Recommended runtime | L4 + High-RAM |
| Tested runtime | About 22 GiB usable VRAM and 53 GiB system RAM |
| Default weights | IQ3_XXS, about 75.8 GB across two GGUF shards |
| RAM requirement | About 43 GB for experts; startup checks for 48 GiB available RAM |
| Default context | 262,144 tokens, int8 KV streaming, 32,768 cells resident in VRAM |
| Full KV cache | About 3.09 GiB of system RAM at 256K, calculated from the engine format |
| Long-input test (previous 128K configuration) | 34,043 tokens, correct recall of an opening marker, about 60 seconds |
| RAM after test (previous 128K configuration) | About 6.4 GiB available |

256K is the base model's native context length. The engine allocates the full host KV cache at startup based on the configured limit; moving from 128K to 256K adds about 1.55 GiB of system RAM. Other buffers also affect total RAM and peak VRAM. The 256K configuration has not been tested on L4, and full 128K and 256K inputs remain untested. The long-input time includes prefill and does not measure decode tps. This deployment serves text only. The first build takes about 10–30 minutes; download time depends on network speed. The disk check reserves the missing download sizes plus 24 GiB of workspace.

## Model and capabilities

[The alesha-pro HF repository](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF) combines unchanged [ISTA-DASLab GSQ-RCO weights](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) with a direction extracted from Huihui. [Strata](https://github.com/Niko1221/Strata) applies projection at runtime, here on layers 4–44 at strength 1.0.

The author tested IQ3_S and IQ2_XS, not IQ3_XXS. Both tested sizes passed five small coding tasks and marker retrieval at 32K/115K. At 32K, decode reached about 59/87 tps on RTX 3090 + EPYC. These results do not establish L4 IQ3_XXS accuracy or speed. See the [measurements](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF#measurements).

Optional IQ2_XS uses about 68 GB of downloads and 35.5 GB of expert RAM; IQ3_S uses about 83.6/50 GB. Prefer IQ3_XXS for a 53 GiB runtime, or IQ2_XS when RAM is tight. Model terms are in the [HF license](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF/blob/main/LICENSE); engine terms are in its upstream repository.

## Use the monthly Pro allowance

Prefer an existing **Colab Pro or eligible Google AI Pro** allowance. Start L4 High-RAM when needed, then disconnect and delete the runtime after use. These subscriptions have different benefits; check your account in Colab.

The reported L4 High-RAM rate for this project is **1.57 compute units/hour**, not a dollar price or a fixed tariff. At that rate, 100 units give about **63.7 hours**, and 200 give about **127.4 hours** in total. Build, download, and idle time also consume units. These totals do not describe a single session's lifetime.

Google lists 100 monthly units for Colab Pro. Google AI benefits and hardware availability can change. Check the live resource panel for your rate and balance. See [official plans](https://developers.google.com/workspace/admin/reseller/v1/how-tos/products#google_colab) and the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Keep a Colab runtime active

The model service runs in the background after startup. To keep notebook execution active while you use it, run this health-check loop in a new cell. It checks the service every five minutes and prints its status. The L4 continues to consume compute units while the loop runs.

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

Return to the cell and press Colab's stop button to interrupt the loop. To stop inference too, run `stop_owned("tunnel")` and `stop_owned("server")`. Colab Pro+ supports continuous execution for up to 24 hours when enough compute units remain. Colab Pro runtime length depends on runtime limits and resource availability; this loop cannot prevent runtime reclamation. See Google's [runtime FAQ](https://research.google.com/colaboratory/faq.html). Do not simulate browser clicks or bypass Colab's idle limits with scripts.

## Option 1: Colab MCP and an agent skill

1. Install uv: `pip install uv`.
2. Configure [googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp) in your local agent. Clients with `mcp.json` support can use:

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

3. Sign in to Google in the browser. Ask the agent to open the MCP connection page. The client must support `notifications/tools/list_changed` to discover notebook tools after connection.
4. Ask the agent to use [skills/strata-colab/SKILL.md](skills/strata-colab/SKILL.md):

   ```text
   Read this project's skills/strata-colab/SKILL.md. Use Colab MCP to import
   strata_l4_gsq_rco_iq3_xxs.ipynb, select L4 High-RAM, run the cells in order,
   test inference and authentication, and supply client configuration.
   Use the default IQ3_XXS weights and 256K context.
   ```

5. For automatic discovery, install the entire `skills/strata-colab` folder in your agent's skills directory. Configure clients with the URL and key from this run.

MCP does not require Chromium CDP. If opening the default browser fails and Chromium already exposes CDP on port 9222, use `python scripts/open_colab_browser.py '<connection URL returned by MCP>'`. This URL contains a private token.

## Option 2: Import into Colab manually

1. Open [Colab](https://colab.research.google.com/). Choose **File → Upload notebook** and upload the `.ipynb`.
2. Choose **Runtime → Change runtime type**, select **L4** and **High-RAM**, then connect.
3. Run the five code cells in order: environment/build, download, pack/configure, start/local tests, tunnel/public tests.
4. Copy the printed Base URL and Model. Prefer to set `STRATA_API_KEY` in Colab Secrets with notebook access enabled.
5. Configure your client. Run the final notebook cell to read the key when needed. Its output contains the secret; do not share or save a notebook with that output. Delete the runtime when finished to stop unit consumption.

The public HF repository needs no token. Set `HF_TOKEN` in Colab Secrets for restricted downloads. API keys require at least 24 ASCII characters with no whitespace. If absent, the notebook generates one in a private runtime file. Keep keys out of shared notebook output.

## Configuration and API

Edit the first code cell before a fresh run. To change context on a running service, update the variables in the current kernel, then rerun pack/configuration and startup. Startup stops the old service recorded by this project before checking memory.

| Variable | Default | Purpose |
|---|---|---|
| `QUANT` | `IQ3_XXS` | Also supports `IQ2_XS`, `IQ3_S` |
| `MAX_CONTEXT` | `262144` | Shared input/output limit |
| `KV_RESIDENT` | `32768` | VRAM window; full KV stays in RAM |
| `PREFILL` | `512` | Prefill batch size |
| `ENABLE_MTP` | `False` | Lookup drafting with `--spec 2`; MTP needs extra downloads and memory |
| `PORT` | `8095` | Avoids Colab's possible 8080 listener; selects 8095–8105 if occupied |

Strata provides OpenAI Chat Completions (`POST /v1/chat/completions`), Anthropic Messages, and OpenAI Responses endpoints. Its [API documentation](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#using-it) lists endpoints, request examples, and authentication. This Pi setup selects `openai-completions`, so it uses the Chat Completions route. Send the key as `Authorization: Bearer <key>`. Add the provider to `~/.pi/agent/models.json`, preserve existing providers, and use:

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
      "contextWindow": 262144,
      "maxTokens": 4096
    }]
  }
}
```

Merge this object under `providers`. Enter the key securely, then start Pi in the same terminal:

```bash
read -rsp 'API key: ' STRATA_API_KEY; printf '\n'
export STRATA_API_KEY
pi
```

Run `/model` and select `strata-colab/alesha-qwen3.8-flash-next-gsq-rco-iq3_xxs`. Pi's `api` field selects the wire format: `openai-completions` sends OpenAI Chat Completions requests to Strata's `/v1/chat/completions`; `baseUrl` ends in `/v1`. Set `contextWindow` to the server's 262144-token limit. `maxTokens` sets the output cap per reply. See Strata's [API reference](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#using-it).

Pi 1.1.0 has no setting to choose a separate compaction model. That requires an extension using `session_before_compact`. In `settings.json`, the model's `compaction.modelOverrides` can use `reserveTokens: 8192` and `keepRecentTokens: 32768`. This compacts history at about 248K. Store critical constraints separately because summaries can lose detail.

The temporary domain can change when the tunnel restarts. The Strata panel's key setting is under **About → Settings**. Long uncached inputs need a longer client timeout.

## Maintenance

The root `.py` is the single source; the `.ipynb` is generated without outputs. `scripts/` contains the builder and optional CDP opener. `tests/` covers configuration, downloads, process ownership, and browser connections. `skills/strata-colab/SKILL.md` supplies the agent workflow.

```bash
python scripts/build_notebook.py
python -m unittest discover -s tests -p 'test_*.py' -q
```

Engine and llama.cpp commits are pinned in the source. Downloads check shard sizes and GGUF headers, without a full-file SHA256 check. Runtime data lives in `/content/Strata-data`; the engine lives in `/content/Strata-colab`. Colab runtime deletion removes these files. This project does not back them up to Drive automatically.
