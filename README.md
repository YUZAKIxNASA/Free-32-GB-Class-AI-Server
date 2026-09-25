# Free 32 GB-Class AI Server on Kaggle — Ollama + Qwen3.8 27B + MTP | @ঔৣ፝ N4!

This notebook turns a free **Kaggle GPU session** into a temporary Ollama server for running a large GGUF model across **2× NVIDIA Tesla T4 GPUs**.

### What this notebook sets up

- Ollama on Kaggle
- Qwen3.8 27B Uncensored `Q4_K_M` from Hugging Face
- Model split across both T4 GPUs
- MTP (speculative decoding) configuration
- 64K context
- Optional Cloudflare Quick Tunnel for a temporary public API
- A simple local/public chat helper and benchmark

### Before you start

In Kaggle, open **Settings → Accelerator** and select **GPU T4 ×2** when available. Also enable **Internet**.

> **VRAM note:** 2× T4 provides 30,720 MiB (about 30 GiB) of physical VRAM. This is a 32 GB-class setup; the notebook verifies the actual GPU memory with `nvidia-smi`.

> **Storage note:** the Kaggle working disk is limited. This notebook downloads the model directly into Ollama's model store and avoids the manual GGUF → `ollama create` route, which would require a second full copy of the ~17 GB model.

## 1. Check the GPU

Run this first. You should see **two Tesla T4 GPUs**.

```python
!nvidia-smi
```

## 2. Install Ollama and the required system package

Run this cell once per fresh Kaggle session.

```python
# Install Ollama
!apt-get update -qq
!apt-get install -y -qq zstd
!curl -fsSL https://ollama.com/install.sh | sh
!which ollama
!ollama --version

```

## 3. Start Ollama with a Kaggle-friendly configuration

The model will be stored in `/kaggle/working/ollama-models` and the server will keep the model loaded after inference.

### 64K Context Configuration
This demo uses a **65,536-token (64K) context window**. With 2× Tesla T4 GPUs, the model is distributed across both GPUs. The notebook will automatically install/locate Ollama if the executable is missing.


```python
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODELS = Path("/kaggle/working/ollama-models")
SOURCE_MODEL = "hf.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M"
MTP_MODEL = "qwen3.8-27b-uncensored-mtp"

# 64K context: larger context window for the live demo.
NUM_CTX = 65536
DRAFT_TOKENS = 2

OLLAMA_MODELS.mkdir(parents=True, exist_ok=True)
os.environ["OLLAMA_MODELS"] = str(OLLAMA_MODELS)

# Make the notebook resilient if Ollama is missing after a fresh/restarted Kaggle session.
OLLAMA_BIN = shutil.which("ollama")
if not OLLAMA_BIN:
    print("Ollama executable not found. Installing it now...")
    subprocess.run(
        ["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"],
        check=True,
    )
    OLLAMA_BIN = shutil.which("ollama")

if not OLLAMA_BIN:
    raise RuntimeError(
        "Ollama installation completed but the executable could not be found."
    )

OLLAMA_ENV = os.environ.copy()
OLLAMA_ENV.update({
    "OLLAMA_HOST": "127.0.0.1:11434",
    "OLLAMA_MODELS": str(OLLAMA_MODELS),
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_CONTEXT_LENGTH": str(NUM_CTX),
    "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_MAX_LOADED_MODELS": "1",
})

def ollama_ready():
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2).close()
        return True
    except Exception:
        return False

if not ollama_ready():
    server_log = open("/tmp/ollama-server.log", "a")
    ollama_server_process = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        env=OLLAMA_ENV,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    for _ in range(60):
        if ollama_ready():
            break
        time.sleep(1)
    else:
        server_log.close()
        raise RuntimeError(
            "Ollama did not become ready. Check /tmp/ollama-server.log"
        )

print("✅ Ollama is ready:", OLLAMA_URL)
print("✅ Ollama binary:", OLLAMA_BIN)
print("✅ Model directory:", OLLAMA_MODELS)
print("✅ Context length:", NUM_CTX)

```

## 4. Download the 27B GGUF directly into Ollama

Do **not** manually download the GGUF and then run `ollama create` from the raw file. That route can require another full copy and exhaust Kaggle's working disk.

The first pull is attempted normally. The `--insecure` fallback is included for Ollama builds that hit the Hugging Face/Xet error `blocked redirect to a different host`. Prefer a fixed Ollama release when your environment has one that resolves the redirect without this fallback.

```python
import subprocess

# Ollama 0.34.x may hit a Hugging Face/Xet redirect error in this environment.
# --insecure is used only as a practical workaround for that redirect.
pull_args = [OLLAMA_BIN, "pull", SOURCE_MODEL]

result = subprocess.run(pull_args, env=OLLAMA_ENV)
if result.returncode != 0:
    print("Standard pull failed; retrying with --insecure...")
    subprocess.run(
        [OLLAMA_BIN, "pull", "--insecure", SOURCE_MODEL],
        env=OLLAMA_ENV,
        check=True,
    )

print("\n✅ Model is available:")
subprocess.run([OLLAMA_BIN, "list"], env=OLLAMA_ENV, check=True)

```

## 5. Confirm the model is installed

```python
!{OLLAMA_BIN} list
```

## 6. Load the base 27B model and verify GPU execution

This request loads the model. Ollama should report `100% GPU` and both T4s should contain model memory.

```python
!{OLLAMA_BIN} run "hf.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M" "Reply with exactly: Model is working."
```

```python
!{OLLAMA_BIN} ps
```

```python
!nvidia-smi
```

## 7. Create the MTP model

This creates an Ollama model definition that reuses the already-downloaded base model and sets the MTP parameters used by this notebook.

```python
import json
import urllib.request

payload = {
    "model": MTP_MODEL,
    "from": SOURCE_MODEL,
    "parameters": {
        "draft_num_predict": DRAFT_TOKENS,
        "num_ctx": NUM_CTX,
    },
    "stream": False,
}
request = urllib.request.Request(
    f"{OLLAMA_URL}/api/create",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.loads(response.read())
print(result)
```

## 8. Load and pin the MTP model

```python
import json
import urllib.request

payload = {
    "model": MTP_MODEL,
    "prompt": "Reply with exactly: MTP is working.",
    "stream": False,
    "keep_alive": -1,
    "options": {
        "draft_num_predict": DRAFT_TOKENS,
        "num_ctx": NUM_CTX,
    },
}
request = urllib.request.Request(
    f"{OLLAMA_URL}/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.loads(response.read())
print("Response:", result.get("response", "").strip())
print("MTP model loaded and pinned.")
```

```python
!{OLLAMA_BIN} ps
```

```python
!nvidia-smi
```

## 9. Benchmark generation speed

This is a simple session-level measurement. Results can vary with prompt, model state, and Kaggle hardware conditions.

```python
import json
import time
import urllib.request

prompt = """Explain in about 200 words why GPUs are useful for running large language models. Cover memory bandwidth, parallel computation, and matrix multiplication."""
payload = {
    "model": MTP_MODEL,
    "prompt": prompt,
    "think": False,
    "stream": False,
    "keep_alive": -1,
    "options": {
        "draft_num_predict": DRAFT_TOKENS,
        "num_ctx": NUM_CTX,
        "temperature": 0,
    },
}
start = time.perf_counter()
request = urllib.request.Request(
    f"{OLLAMA_URL}/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.loads(response.read())
wall_time = time.perf_counter() - start
eval_count = result.get("eval_count", 0)
eval_seconds = result.get("eval_duration", 0) / 1e9
speed = eval_count / eval_seconds if eval_seconds else 0
print(result.get("response", "").strip())
print("\n--- Benchmark ---")
print(f"Generated tokens: {eval_count}")
print(f"Generation time: {eval_seconds:.2f}s")
print(f"Wall time: {wall_time:.2f}s")
print(f"Generation speed: {speed:.2f} tokens/s")
```

# Optional: expose Ollama through a temporary public URL

A Cloudflare Quick Tunnel gives you a temporary HTTPS endpoint such as `https://xxxxx.trycloudflare.com`.

**Security:** the quick tunnel has no authentication. Anyone who gets the URL can send requests to your Ollama server while the Kaggle session and tunnel are alive. Do not use this for sensitive data and do not publish the URL in a video description or public post.

## 10. Install Cloudflare Tunnel

Skip this section if you only need the local Ollama API inside Kaggle.

```python
!curl -fL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb" -o /tmp/cloudflared-linux-amd64.deb
!dpkg -i /tmp/cloudflared-linux-amd64.deb
!cloudflared --version
```

## 11. Start the Cloudflare Quick Tunnel

```python
import queue
import re
import subprocess
import threading
import time

def pipe_process_output(process, output_queue):
    for line in iter(process.stdout.readline, ""):
        output_queue.put(line)

tunnel_is_running = (
    "cloudflared_process" in globals()
    and cloudflared_process.poll() is None
    and "PUBLIC_OLLAMA_URL" in globals()
    and PUBLIC_OLLAMA_URL
)

if not tunnel_is_running:
    cloudflared_process = subprocess.Popen(
        [
            "cloudflared", "tunnel",
            "--no-autoupdate",
            "--protocol", "http2",
            "--edge-ip-version", "4",
            "--url", OLLAMA_URL,
            "--http-host-header", "localhost:11434",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_queue = queue.Queue()
    threading.Thread(
        target=pipe_process_output,
        args=(cloudflared_process, output_queue),
        daemon=True,
    ).start()
    deadline = time.monotonic() + 45
    PUBLIC_OLLAMA_URL = None
    while time.monotonic() < deadline:
        if cloudflared_process.poll() is not None and output_queue.empty():
            break
        try:
            line = output_queue.get(timeout=1)
        except queue.Empty:
            continue
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if match:
            PUBLIC_OLLAMA_URL = match.group(0)
            break
    if not PUBLIC_OLLAMA_URL:
        raise RuntimeError("Cloudflare did not produce a public URL. Rerun this cell.")

print("Public Ollama API:", PUBLIC_OLLAMA_URL)
print("OpenAI-compatible base URL:", f"{PUBLIC_OLLAMA_URL}/v1")
print("Model name:", MTP_MODEL)
```

## 12. Test the Ollama API locally or through the tunnel

```python
import json
import urllib.request

def chat(prompt, base_url=OLLAMA_URL, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": MTP_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "draft_num_predict": DRAFT_TOKENS,
            "num_ctx": NUM_CTX,
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
    return result["message"]["content"]

print(chat("Write a one-line Python hello-world program."))
```

### Test through the public tunnel

Run this after the Cloudflare tunnel cell prints `PUBLIC_OLLAMA_URL`.

```python
print(chat("Hello from the Cloudflare tunnel.", base_url=PUBLIC_OLLAMA_URL))
```

## Optional: OpenAI-compatible client settings

Use the printed URL with `/v1` as the base URL in OpenAI-compatible tools.

- **Base URL:** `https://YOUR-TUNNEL.trycloudflare.com/v1`
- **Model:** `qwen3.8-27b-uncensored-mtp`
- **API key:** a placeholder value accepted by your client

The tunnel URL is temporary and normally disappears when the Kaggle session or tunnel stops.

## Interactive Ollama CLI (optional)

Run this only after all setup cells are complete.

```python
!{OLLAMA_BIN} run qwen3.8-27b-uncensored-mtp
```
