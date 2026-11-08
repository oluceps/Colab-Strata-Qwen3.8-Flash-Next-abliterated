# %% [markdown]
# # Strata / GSQ-RCO on Colab L4 / A100
# 按顺序运行。首次编译可能需要 10–30 分钟，下载时间取决于连接速度。
# IQ3_XXS 的 128K 配置已在 L4（约 22 GiB 可用显存、53 GiB 系统内存）通过启动与短请求测试。
# 默认上下文 256K，使用 k8v4 KV streaming。主模型完整 KV 约占 2.39 GiB RAM，尚未在 L4 实测。
# IQ3_S：下载约 83.6 GB，专家占约 50 GB RAM。IQ2_XS：约 68 GB / 35.5 GB。
# IQ3_XXS：下载约 75.8 GB，专家占约 43 GB RAM。仓库作者未实测此档位。
# IQ3_S 全新安装建议约 110 GB 可用磁盘。代码按缺失文件总量加 24 GiB 余量检查。
# 64 GB 是机器内存档位，不是必须剩余 64 GiB。启动前另查可用内存。
# README 测速来自 RTX 3090 + EPYC，不代表 L4 的速度。
# 此配置只提供文本服务，不下载视觉投影器。
# 保留已有模型文件。新引擎和 pack 使用独立目录，不覆盖旧安装。
# 默认启用 MTP（--spec 4）与 lookup chain，自动预填充，先测试本地推理，再开启公网隧道。
# 来源：https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF
# 权重保持原样。控制向量在运行时投影，层范围 4–44，强度 1.0。

# %%
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

STRATA_REV = "d5ea7133741e67743c0e886bb426c0ce8d69cf6c"
LLAMA_REV = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
REPO_ID = "alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF"
QUANT = "IQ3_XXS"  # 可选 IQ3_S 或 IQ2_XS。默认适配约 53 GiB RAM 的 L4 运行时。
if QUANT not in {"IQ3_S", "IQ3_XXS", "IQ2_XS"}:
    raise ValueError("此 notebook 支持 IQ3_S、IQ3_XXS 和 IQ2_XS。")
# 这是启动前的可用内存预算，不是作者保证的最低硬件规格。
MIN_AVAILABLE_GIB = {"IQ3_S": 54, "IQ3_XXS": 48, "IQ2_XS": 40}[QUANT]
ROOT = Path("/content/Strata-colab")
DATA = Path("/content/Strata-data")
MODEL_DIR = DATA / "models/alesha-gsq-rco"
PACK = DATA / f"packs/alesha-{QUANT.lower()}-colab"
STATE = DATA / f"colab-alesha-{QUANT.lower()}"
PORT = 8095  # Colab 的 node 服务可能占用 8080。
MAX_CONTEXT = 262144
KV_TYPE = "k8v4"  # 8-bit Key + 4-bit Value；可切回 "int8" 比较精度与速度。
KV_RESIDENT = 32768  # KV streaming：完整缓存放在 RAM，窗口留在显存。
PREFILL = "auto"  # 按显存预算选择分块，默认上限 8192；可设为 1024/2048 做对照。
ENABLE_MTP = True  # 主模型验证草稿；设为 False 可对比仅 lookup 的速度与显存。
DOWNLOAD_WORKERS = 2  # 并行下载两个大分片；每个文件由 HF Xet 并发获取数据块。
XET_HIGH_PERFORMANCE = True  # 高性能模式；下载数据缓冲默认限制在 4 GiB。
SHARDS = [f"{QUANT}/Qwen3.8-Flash-Next-GSQ-RCO-{QUANT}-{i:05d}-of-00002.gguf" for i in (1, 2)]
VECTOR_NAME = "Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf"
VECTOR = MODEL_DIR / VECTOR_NAME

def run(args, cwd=None, env=None):
    child_env = dict(os.environ if env is None else env, PYTHONUNBUFFERED="1")
    with subprocess.Popen([str(x) for x in args], cwd=cwd, env=child_env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", bufsize=1) as proc:
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if code:
        raise RuntimeError(f"子进程退出码：{code}。具体错误见上方输出。")

def capture(args):
    return subprocess.check_output([str(x) for x in args], text=True).strip()

def available_gib():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) / 1024**2

if not Path("/content").is_dir():
    raise RuntimeError("请在 Colab GPU 运行时执行此 notebook。")
gpu = capture(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap", "--format=csv,noheader,nounits"])
print(gpu)
if len(gpu.splitlines()) != 1:
    raise RuntimeError("此 notebook 需要单卡 NVIDIA GPU。")
gpu_name, vram_mib, free_vram_mib, capability = [v.strip() for v in gpu.split(",")]
CUDA_ARCH = capability.replace(".", "")
if not CUDA_ARCH.isdigit() or int(CUDA_ARCH) < 80 or float(vram_mib) < 11000:
    raise RuntimeError("此配置需要 Ampere 或更新架构、12 GB 档位以上显存。推荐 L4 24 GB。")
if available_gib() < MIN_AVAILABLE_GIB:
    raise RuntimeError(f"{QUANT} 需要至少 {MIN_AVAILABLE_GIB} GiB 可用系统内存。请选择高内存运行时。")
print(f"可用系统内存：{available_gib():.1f} GiB")
print(f"可用磁盘：{shutil.disk_usage('/content').free / 2**30:.1f} GiB")
NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
run([NVCC, "--version"])
for path in (MODEL_DIR, STATE):
    path.mkdir(parents=True, exist_ok=True)

def checkout(url, dest, revision):
    if not (dest / ".git").exists():
        if dest.exists() and any(dest.iterdir()):
            raise RuntimeError(f"目录不是 Git 仓库：{dest}")
        run(["git", "init", dest])
        run(["git", "-C", dest, "remote", "add", "origin", url])
    if capture(["git", "-C", dest, "remote", "get-url", "origin"]) != url:
        raise RuntimeError(f"仓库来源不匹配：{dest}")
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.stdout.strip() != revision:
        if capture(["git", "-C", dest, "status", "--porcelain"]):
            raise RuntimeError(f"仓库有未提交文件，请保留文件后再处理：{dest}")
        run(["git", "-C", dest, "fetch", "--depth", "1", "origin", revision])
        run(["git", "-C", dest, "checkout", "--detach", revision])
    elif capture(["git", "-C", dest, "diff", "HEAD", "--"]):
        raise RuntimeError(f"源码已修改：{dest}")

checkout("https://github.com/Niko1221/Strata.git", ROOT, STRATA_REV)
PY = ROOT / ".venv/bin/python"
if not PY.exists():
    run([sys.executable, "-m", "pip", "install", "uv"])
    run([sys.executable, "-m", "uv", "venv", "--seed", "--python", sys.executable, ROOT / ".venv"])
run([PY, "-m", "pip", "install", "-r", ROOT / "requirements.txt",
     "huggingface_hub==2.4.0", "hf-xet==1.8.0"])
LLAMA = ROOT / "third_party/llama.cpp"
checkout("https://github.com/ggml-org/llama.cpp.git", LLAMA, LLAMA_REV)
ENV = dict(os.environ, STRATA_GGUF_PY=str(LLAMA / "gguf-py"))
ENV["PATH"] = str(PY.parent) + os.pathsep + ENV.get("PATH", "")
BUILD = ROOT / f"build-colab-sm{CUDA_ARCH}"
run([PY.parent / "cmake", "-S", ROOT, "-B", BUILD, "-G", "Ninja",
     "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
     "-DSTRATA_NATIVE_EXPERTS=ON", f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}",
     f"-DCMAKE_CUDA_COMPILER={NVCC}", f"-DSTRATA_GGML_DIR={LLAMA}"], env=ENV)
with (STATE / "build.log").open("w") as log:
    print(f"编译日志：{log.name}")
    result = subprocess.run([str(PY.parent / "cmake"), "--build", str(BUILD), "--target", "strata",
                             "-j", str(min(4, os.cpu_count() or 2))], env=ENV, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    print((STATE / "build.log").read_text(errors="replace")[-12000:])
    raise RuntimeError("引擎编译失败。请检查上方日志。")
print(f"{gpu_name} sm_{CUDA_ARCH} 引擎编译完成。")

# %% [markdown]
# ## 下载并检查两个分片与控制向量
# 公共模型不要求登录。受限模型可使用 Colab Secret `HF_TOKEN`。
# 保留旧 Orca 文件，但不用于本模型。下载失败后可以重跑此单元。
# 每 30 秒显示下载目录占用空间，避免长时间无输出。

# %%
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        hf_token = None
download_env = dict(ENV)
download_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
download_env["HF_XET_HIGH_PERFORMANCE"] = "1" if XET_HIGH_PERFORMANCE else "0"
# 新版 Xet 的内存预算可显式限制；保留用户提供的覆盖值。
download_env.setdefault("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE", str(2 * 2**30))
download_env.setdefault("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE", str(512 * 2**20))
download_env.setdefault("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT", str(4 * 2**30))
if hf_token:
    download_env["HF_TOKEN"] = hf_token
download_script = r'''
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, os, shutil, threading, time
repo, destination, names, workers = __import__('sys').argv[1:]
destination = Path(destination)
names = json.loads(names)
workers = max(1, min(int(workers), len(names)))
token = os.environ.get("HF_TOKEN") or False
api = HfApi(token=token)
print('读取 Hugging Face 模型信息……', flush=True)
info = api.model_info(repo, files_metadata=True)
files = {f.rfilename: f for f in info.siblings}
missing = [n for n in names if n not in files]
if missing:
    raise RuntimeError(f"模型仓库缺少分片：{missing}")
for name in names:
    if not files[name].size:
        raise RuntimeError(f"服务器没有返回文件大小：{name}")
pending = sum(files[n].size for n in names
              if not (destination / n).is_file() or (destination / n).stat().st_size != files[n].size)
# 一次为所有并行任务预留空间；不把其他线程正在写入的文件当作额外下载预算。
if shutil.disk_usage(destination).free < pending + 24 * 2**30:
    raise RuntimeError(f"需要 {pending / 2**30 + 24:.1f} GiB 可用磁盘，包含安装余量。")
print(f'并行文件数：{workers}；待下载文件总大小：{pending / 2**30:.2f} GiB', flush=True)

def fetch_and_verify(name):
    size = files[name].size
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size != size:
        if target.is_symlink():
            raise RuntimeError(f"旧模型分片不完整：{target}")
        print(f'下载或重试：{name}', flush=True)
        hf_hub_download(repo_id=repo, filename=name, revision=info.sha,
                        local_dir=destination, token=token)
    else:
        print(f'复用完整文件：{name}', flush=True)
    if target.stat().st_size != size:
        raise RuntimeError(f"模型分片大小不符：{name}")
    with target.open('rb') as f:
        if f.read(4) != b'GGUF':
            raise RuntimeError(f"文件不是 GGUF：{name}")
    return {"name": name, "size": size, "mtime_ns": target.stat().st_mtime_ns}

stop = threading.Event()
def progress():
    while not stop.wait(30):
        size_now = 0
        for path in destination.rglob('*'):
            try:
                if path.is_file():
                    size_now += path.stat().st_size
            except FileNotFoundError:
                pass  # Xet 可能正在把临时文件重命名为完成文件。
        print(f'下载仍在运行，目录文件共 {size_now / 2**30:.2f} GiB', flush=True)
worker = threading.Thread(target=progress, daemon=True)
worker.start()
try:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(fetch_and_verify, names))
finally:
    stop.set()
    worker.join()
manifest = destination / 'colab-download.json'
tmp = manifest.with_suffix('.tmp')
tmp.write_text(json.dumps({"revision": info.sha, "files": records}, indent=2))
tmp.replace(manifest)
print('两个分片与控制向量的大小和 GGUF 文件头检查通过。未执行全文件 SHA256 检查。')
'''
try:
    run([PY, "-c", download_script, REPO_ID, MODEL_DIR, json.dumps(SHARDS + [VECTOR_NAME]), DOWNLOAD_WORKERS], env=download_env)
finally:
    download_env.pop("HF_TOKEN", None)
    hf_token = None
GGUF = MODEL_DIR / SHARDS[0]
PLE_GGUF = MODEL_DIR / SHARDS[1]

# %% [markdown]
# ## 转换 pack，并生成配置
# 完成标记包含源码版本和分片信息。中断的 pack 不会被当作完成。
# 不生成额外的 experts.bin，避免复制约 50 GiB 专家权重。
# GSQ-RCO 使用原生张量，不启用 Orca 的 --compat-bf16 转换。
# 默认构建 MTP 草稿层；主模型仍用 IQ3_XXS，KV 使用上方配置，原有采样参数保持不变。

# %%
signature = {"strata": STRATA_REV, "download": json.loads((MODEL_DIR / "colab-download.json").read_text())}
required = ["index.txt", "dense.bin", "native_experts.txt", "conversions.json",
            "tokenizer/vocab.json", "tokenizer/chat_template.jinja"]
marker = PACK / "colab-complete.json"
complete = (marker.is_file() and json.loads(marker.read_text()) == signature
            and all((PACK / x).is_file() and (PACK / x).stat().st_size for x in required))
if not complete:
    if shutil.disk_usage(DATA).free < 8 * 2**30:
        raise RuntimeError("转换前需要至少 8 GiB 可用磁盘空间。")
    PACK.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    run([PY, ROOT / "tools/iq_pack.py", "--gguf", GGUF, "--out", PACK], env=ENV)
    if not all((PACK / x).is_file() and (PACK / x).stat().st_size for x in required):
        raise RuntimeError("pack 缺少必要文件。")
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps(signature, indent=2))
    tmp.replace(marker)
print("pack 完成检查通过。")

MTP = DATA / "mtp-colab"
if ENABLE_MTP:
    run([PY, ROOT / "tools/mtp_fetch.py", "fetch", "--out", MTP], cwd=ROOT, env=ENV)
    run([PY, ROOT / "tools/mtp_pack.py", "--src", MTP, "--experts", "q2_0",
         "--out", MTP / "mtp-q2_0.gguf"], cwd=ROOT, env=ENV)
    run([PY, ROOT / "tools/mtp_rt.py", "--gguf", MTP / "mtp-q2_0.gguf",
         "--out", MTP / "rt"], cwd=ROOT, env=ENV)
    shutil.copy2(ROOT / "data/draft_vocab.bin", MTP / "rt/draft_vocab.bin")

args = ["--pack", str(PACK), "--native", str(GGUF), "--ple-gguf", str(PLE_GGUF),
        "--expert-profile", str(ROOT / "data/expert-profile.bin"), "--expert-cache", "auto",
        "--prefill", str(PREFILL), "--spec", "4" if ENABLE_MTP else "2",
        "--max-context", str(MAX_CONTEXT), "--kv", str(KV_TYPE),
        "--adapt-every", "1", "--adapt-swaps", "96", "--prompt-cache-tail"]
if KV_RESIDENT and MAX_CONTEXT > KV_RESIDENT:
    args += ["--kv-resident", str(KV_RESIDENT)]
if ENABLE_MTP:
    args += ["--mtp", str(MTP / "rt"), "--spec-min-p", "0.5", "--lookup-chain", "4"]
args += ["--control-vector-scaled", f"{VECTOR}:1.0",
         "--control-vector-layer-range", "4", "44", "--cvec-mode", "project", "--cvec-dir", "per-layer"]
MODEL_NAME = f"alesha-qwen3.8-flash-next-gsq-rco-{QUANT.lower()}"
CONFIG = STATE / "strata-alesha.json"
config = {"exe": str(BUILD / "strata"), "args": args, "cwd": str(ROOT),
          "tokenizer": str(PACK / "tokenizer"), "model_name": MODEL_NAME,
          "sampling": {"experimental_speed_projection": True},
          "expert_profile_save": str(STATE / "expert-profile-learned.bin"),
          "expert_profile_save_every": 5,
          "log": str(STATE / "engine.log"), "host": "127.0.0.1", "port": PORT}
CONFIG.write_text(json.dumps(config, indent=2))
print(f"配置：context={MAX_CONTEXT}, kv={KV_TYPE}, prefill={PREFILL}, MTP={ENABLE_MTP}")

# %% [markdown]
# ## 启动服务，并测试推理
# API key 使用 Colab Secret `STRATA_API_KEY`，或自动生成并保存在运行时文件中。
# 只重启此 notebook 创建且进程身份匹配的服务。
# 端口被占用时自动选择空闲端口，并同步配置与后续隧道。

# %%
import secrets
import signal
import errno

def select_available_port(preferred):
    candidates = list(dict.fromkeys([preferred, *range(8095, 8106)]))
    for candidate in candidates:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", candidate))
                return candidate
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
    raise RuntimeError("8095–8105 均被占用。请设置其他 PORT。")

def process_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None

def stop_owned(name):
    record = STATE / f"{name}.pid.json"
    if not record.exists():
        return
    meta = json.loads(record.read_text())
    pid = meta["pid"]
    if process_identity(pid) == meta["start"]:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            record.unlink(missing_ok=True)
            return
        for _ in range(40):
            if process_identity(pid) != meta["start"]:
                break
            time.sleep(0.25)
        else:
            if process_identity(pid) == meta["start"]:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    record.unlink(missing_ok=True)

def spawn(name, command, env):
    with (STATE / f"{name}.log").open("w") as log:
        proc = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (STATE / f"{name}.pid.json").write_text(json.dumps({"pid": proc.pid, "start": process_identity(proc.pid)}))
    return proc

def request(path, payload=None, authenticated=True, base=None, timeout=15):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = "Bearer " + API_KEY
    req = urllib.request.Request((base or f"http://127.0.0.1:{PORT}") + path,
                                 data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

SMOKE_OK = None
stop_owned("server")
old_port = PORT
PORT = select_available_port(PORT)
if PORT != old_port:
    stop_owned("tunnel")
    print(f"端口 {old_port} 被占用，服务改用 {PORT}。")
config = json.loads(CONFIG.read_text())
config["port"] = PORT
CONFIG.write_text(json.dumps(config, indent=2))
print(f"本地 API：http://127.0.0.1:{PORT}/v1")
if available_gib() < MIN_AVAILABLE_GIB:
    raise RuntimeError(f"当前只有 {available_gib():.1f} GiB 可用内存，{QUANT} 的启动检查要求 {MIN_AVAILABLE_GIB} GiB。")
API_KEY = os.environ.get("STRATA_API_KEY")
if not API_KEY:
    try:
        from google.colab import userdata
        API_KEY = userdata.get("STRATA_API_KEY")
    except Exception:
        API_KEY = None
key_path = STATE / "api-key.txt"
if not API_KEY:
    API_KEY = key_path.read_text().strip() if key_path.exists() else secrets.token_urlsafe(32)
API_KEY = API_KEY.strip()
if len(API_KEY) < 24 or not API_KEY.isascii() or any(c.isspace() for c in API_KEY):
    raise RuntimeError("请使用至少 24 位、不含空白的 ASCII API key。")
key_path.touch(mode=0o600, exist_ok=True)
key_path.chmod(0o600)
key_path.write_text(API_KEY)
server_env = dict(ENV, STRATA_API_KEY=API_KEY)
server = spawn("server", [PY, "-m", "serve.server", "--engine", "strata", "--config", CONFIG,
                          "--host", "127.0.0.1", "--port", PORT], server_env)
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    if server.poll() is not None:
        print((STATE / "server.log").read_text(errors="replace")[-12000:])
        raise RuntimeError(f"服务退出。请检查 {STATE / 'engine.log'}。")
    try:
        health = request("/health")
        if health.get("status") == "ok" and health.get("loaded") and health.get("model") == MODEL_NAME:
            break
    except (OSError, ValueError):
        pass
    time.sleep(3)
else:
    raise TimeoutError(f"服务在 15 分钟内未就绪。进程保留，日志：{STATE / 'server.log'}")
models = request("/v1/models")
assert any(x["id"] == MODEL_NAME for x in models["data"]), models
try:
    request("/v1/models", authenticated=False)
except urllib.error.HTTPError as exc:
    if exc.code != 401:
        raise
else:
    raise RuntimeError("API 未拒绝匿名请求。停止公网发布。")
smoke_payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "What is 17 + 25? Reply with only the number."}],
                 "max_tokens": 32, "temperature": 0, "reasoning_effort": "none"}
t0 = time.monotonic()
answer = request("/v1/chat/completions", smoke_payload, timeout=300)
content = answer["choices"][0]["message"].get("content") or ""
if content.strip().rstrip(".") != "42":
    raise RuntimeError(f"推理测试未通过，返回：{content!r}")
SMOKE_OK = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
print(f"本地推理通过：{content.strip()}；请求耗时 {time.monotonic() - t0:.2f} 秒")
print("API 鉴权检查通过。")

# %% [markdown]
# ## 创建公网隧道
# 仅在本地推理和鉴权测试通过后运行。临时域名可能在重启后变化。
# API key 不自动显示。客户端需要时，在私有单元运行 `print(API_KEY)`。
# Colab 断开或回收运行时后，服务与隧道会停止。

# %%
if globals().get("SMOKE_OK") != hashlib.sha256(CONFIG.read_bytes()).hexdigest() or server.poll() is not None:
    raise RuntimeError("请先通过当前配置的本地推理测试。")
assert request("/health").get("loaded")
stop_owned("tunnel")
CF = STATE / "cloudflared"
if not CF.is_file():
    with urllib.request.urlopen("https://api.github.com/repos/cloudflare/cloudflared/releases/latest", timeout=30) as r:
        release = json.load(r)
    asset = next(x for x in release["assets"] if x["name"] == "cloudflared-linux-amd64")
    digest = asset.get("digest", "")
    if not digest.startswith("sha256:"):
        raise RuntimeError("发布资产没有 SHA256。停止下载。")
    temp = CF.with_suffix(".tmp")
    with urllib.request.urlopen(asset["browser_download_url"], timeout=120) as source, temp.open("wb") as target:
        shutil.copyfileobj(source, target)
    if hashlib.sha256(temp.read_bytes()).hexdigest() != digest.split(":", 1)[1]:
        raise RuntimeError("cloudflared 校验失败。")
    temp.chmod(0o700)
    temp.replace(CF)
    (STATE / "cloudflared-release.json").write_text(json.dumps({"tag": release["tag_name"], "sha256": digest}))
tunnel = spawn("tunnel", [CF, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"], ENV)
PUBLIC_URL = None
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    if tunnel.poll() is not None:
        raise RuntimeError(f"隧道退出，请检查 {STATE / 'tunnel.log'}。")
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", (STATE / "tunnel.log").read_text(errors="replace"))
    if match:
        PUBLIC_URL = match.group(0)
        try:
            external_models = request("/v1/models", base=PUBLIC_URL)
            if any(x["id"] == MODEL_NAME for x in external_models.get("data", [])):
                break
        except (OSError, ValueError):
            pass
    time.sleep(3)
else:
    raise TimeoutError(f"公网 API 尚未就绪。日志：{STATE / 'tunnel.log'}")
try:
    request("/v1/models", authenticated=False, base=PUBLIC_URL)
except urllib.error.HTTPError as exc:
    if exc.code != 401:
        stop_owned("tunnel")
        raise
else:
    stop_owned("tunnel")
    raise RuntimeError("公网 API 未拒绝匿名请求，隧道已停止。")
external = request("/v1/chat/completions", smoke_payload, base=PUBLIC_URL, timeout=90)
if (external["choices"][0]["message"].get("content") or "").strip().rstrip(".") != "42":
    stop_owned("tunnel")
    raise RuntimeError("公网推理测试未通过，隧道已停止。")
print("公网推理与鉴权测试通过。")
print("Base URL:", PUBLIC_URL + "/v1")
print("Model:", MODEL_NAME)
print("API key 文件：", key_path)

# %% [markdown]
# ## 查看日志或停止服务
# 查看日志：`print((STATE / 'engine.log').read_text()[-8000:])`
# 停止服务：先运行 `stop_owned('tunnel')`，再运行 `stop_owned('server')`。
# 要比较 MTP 性能，请保存本次耗时，再设 ENABLE_MTP=False 并重跑配置和测试。
# 请求耗时包含预填充与 HTTP 开销，不等于引擎解码 tokens/s。

# %% [markdown]
# ## 获取 API key
# 仅在需要连接客户端时运行下一个单元。
# 输出包含密钥。请勿分享此单元的输出或保存了该输出的 notebook。

# %%
!cat /content/Strata-data/colab-alesha-iq3_xxs/api-key.txt
