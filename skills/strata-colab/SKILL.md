---
name: strata-colab
description: Deploy and troubleshoot this project's Strata GSQ-RCO IQ3_XXS notebook on Colab L4 High-RAM through Colab MCP, then configure an authenticated chat client.
---

# Strata on Colab L4

Use the user's project checkout. Read `README.md` or `README.en.md` there for current defaults and usage. The source is `strata_l4_gsq_rco_iq3_xxs.py`; the upload artifact is its `.ipynb`. If this skill is installed separately, locate the checkout before editing. Do not assume relative paths from the installed skill lead to the project.

## Connect and deploy

- Use the installed Google Colab MCP connection tool to open the notebook. It needs a local client with dynamic tool-list notifications. After connecting, discover the notebook tools again.
- If sign-in is required, ask the user to sign in in the browser. If tools remain unavailable, inspect the connection and client notification support before changing notebook code.
- A local Chromium CDP endpoint is an optional browser-opening fallback. The project's `scripts/open_colab_browser.py` preserves the MCP URL fragment and bypasses proxy routing for localhost. Use it only when the user has enabled CDP. Keep the connection URL private.
- Import the generated notebook or update the matching cells in an existing notebook. Keep the user's unrelated cells. Save changes before execution.
- Select L4 with High-RAM. Read GPU, available RAM, and disk checks from the first cell. A T4 fails this configuration's architecture requirement. Do not silently change to A100 or bypass memory checks.
- Execute the five code cells sequentially. Inspect each result before continuing. Avoid duplicate execution of a running download or build. A timeout in the MCP call does not prove that runtime work stopped; inspect cell state and logs.

## Deployment invariants

- Default IQ3_XXS uses 256K context, k8v4 KV streaming, and a 32K VRAM window. The main-model host KV cache is about 2.39 GiB at this limit, excluding the draft layer's int8 KV. K8V4 uses 8-bit keys and 4-bit values; set `KV_TYPE="int8"` for precision comparisons. Require 48 GiB available RAM before startup. Memory figures use GiB; HF file sizes use decimal GB.
- Use both GGUF shards. `--native` points to shard 1; `--ple-gguf` points to shard 2, which contains the n-gram table.
- Pack GSQ weights with the notebook's native packing path. Do not add `--compat-bf16`.
- Apply the included control vector at strength 1.0, layers 4–44, `project` mode and `per-layer` directions. The model weights remain unchanged.
- MTP is enabled by default with `--spec 4 --spec-min-p 0.5 --lookup-chain 4`; the total verification window may reach 8 tokens. Every draft is verified by the IQ3_XXS main model. Q2_0 draft experts do not change the main-model quantization. MTP needs about 6 GB of source downloads plus runtime files and additional VRAM/RAM. With `ENABLE_MTP=False`, use lookup drafting at `--spec 2` without MTP-only flags.
- Use `PREFILL="auto"` (up to 8192 by default), `--adapt-every 1 --adapt-swaps 96`, and `--prompt-cache-tail`. Keep `--expert-cache auto`, configured `KV_TYPE` (k8v4 by default), and automatic CPU worker selection. Retain existing sampling settings.
- Save the learned expert profile to `STATE / "expert-profile-learned.bin"` with `expert_profile_save_every: 5`. Saves are checked between requests and occur on a clean exit; the next start reuses the file. It is local to the Colab runtime and is lost when that runtime is deleted.
- Respect pinned Strata and llama.cpp revisions. Update pins only for an identified compatibility need and rerun relevant checks.
- Download up to `DOWNLOAD_WORKERS=2` files concurrently using pinned Hub 2.4.0 / Xet 1.8.0. Xet high-performance mode is enabled, with default download data-buffer budgets of 2 GiB shared, 512 MiB per file, and a 4 GiB total buffer limit; these do not cap total process RAM. Keep the all-files disk preflight, resolved revision, size/header validation, and atomic completion manifest. Reuse complete files; do not promise partial-file resume across processes with current Xet. `DOWNLOAD_WORKERS=1` and `XET_HIGH_PERFORMANCE=False` are available for comparison. Do not add obsolete hf_transfer, range-get, or chunk-cache settings.
- Reuse complete downloads and completed packs through their actual validation, not directory existence alone. A missing or incomplete shard needs repair.
- Stop only processes verified by the notebook's PID and start-time records. Port 8080 can belong to Colab's node service. Use the notebook's port selection instead of broad `pkill` or `fuser -k` commands.

## Check and configure

Require `/health` to report the selected model, `loaded: true`, and the configured context limit. Require an authenticated `/v1/models` call, anonymous rejection with HTTP 401, and the notebook's arithmetic inference check. Publish a tunnel only after these local checks pass, then repeat inference and authentication through the tunnel.

Use Colab Secrets for `STRATA_API_KEY` and optional `HF_TOKEN`. Do not paste keys into source or shared outputs. The notebook generates a private key file when no key is supplied. The last notebook cell can display the saved key when the user needs it. Warn that its output is secret and must not be shared. Give the client the current `/v1` Base URL and model ID; help transfer the key privately.

For an active session, an optional notebook cell can poll `/health` every five minutes while it sleeps. Explain that the running cell continues to consume compute units and cannot guarantee the runtime will avoid reclamation. Colab Pro+ supports continuous execution up to 24 hours when units remain; ordinary Pro is still subject to its runtime limits. Do not use browser click simulation or scripts that bypass Colab idle limits. Stop the health-check cell with Colab's stop button. Stop the tunnel and model server separately when the user wants to end service use.

For Pi, merge the provider into `~/.pi/agent/models.json` without replacing unrelated providers. Explain that `api: openai-completions` selects the OpenAI Chat Completions wire format and maps to `/v1/chat/completions`; Strata also documents Anthropic Messages and OpenAI Responses endpoints. Link to Strata's `docs/DETAILS.md#using-it` API reference. Use `contextWindow: 262144` and `maxTokens: 4096`. Prefer an exported key variable and `apiKey: "$STRATA_API_KEY"`. If context compaction is requested, use the model-specific budget from the README. Report that summaries can lose detail.

Changing context in a loaded runtime requires updating the live variables, regenerating configuration, then restarting through the startup cell. Avoid rerunning environment setup while the model occupies RAM. Preserve a working tunnel when its port stays the same; report a changed domain if a new tunnel is needed.

Report the tested context length, free RAM, and engine decode rate separately from HTTP duration. Decode rates reach 32–57 tps with peaks above 80 tps. Current L4 evidence comes from the previous 128K configuration, including a 34,043-token recall check. The 256K/k8v4 configuration, the new MTP/automatic-prefill defaults, and full 128K/256K inputs remain untested. Measure actual draft acceptance and cache slots; MTP and more frequent expert swaps can add overhead. Do not present HF's RTX 3090 results as L4 IQ3_XXS measurements.

## Troubleshoot

Inspect logs under `/content/Strata-data/colab-alesha-iq3_xxs` for build, engine, server, and tunnel failures. Read the failing cell's output before changing code.

For memory failure, report the available RAM and quantization requirement. Preserve IQ3_XXS and the user-selected KV precision (currently k8v4). First reduce the prefill chunk to 1024/2048 or disable MTP for comparison; use a smaller context only when compatible with the user's goal. Change model or KV precision only when the user explicitly accepts that tradeoff. After a concrete fix, retry the failed stage once and inspect the result. If the same failure persists, report the evidence and resolve the cause before retrying again.

For panel authentication, the API key setting is under About → Settings. For an API timeout on long input, inspect engine progress and client timeout before restarting. Runtime reclamation removes temporary files and invalidates the service.

Update the local source when changing deployment behavior. Run `python scripts/build_notebook.py` and the README's test command before importing the changed artifact. Explain any live restart and its brief interruption. Delete the Colab runtime only when the user requests shutdown or the deployment task includes it.
