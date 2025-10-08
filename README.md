
---

# 使用文档｜RAG（LangChain + Qwen2.5-7B-Instruct + BGE-M3 + Chroma）

适配环境：Linux/集群（你当前的服务器）；GPU：**RTX 3090 24GB（FP16，无量化）**。
参考文章：https://mp.weixin.qq.com/s/GuljXiWyNDfLrESkS2IMRQ

## 1. 目录结构与文件说明

```
your_project/
├─ rag_rs.py                  # 你贴的主程序（FP16，device_map="auto"）
├─ documents/                 # 放你的遥感知识文档（TXT/PDF/DOCX）
└─ vector_db_qwen_bge_m3/     # 向量库目录（运行后自动生成/更新）
```

* `documents/`：随时往里加/删文档；变更后用 `--rebuild` 重建向量库。
* `vector_db_qwen_bge_m3/`：Chroma 的持久化数据目录。

---

## 2. 环境准备

> 建议用独立的 conda 环境，避免和系统包冲突。

```bash
conda create -n ragenv python=3.10 -y
conda activate ragenv

# PyTorch（3090 一定要装 2.6+，满足 transformers 的安全要求）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# LangChain 及相关
pip install -U langchain langchain-community langchain-huggingface langchain-text-splitters

# 向量库 & 工具
pip install chromadb sentence-transformers python-dotenv pypdf python-docx

# Transformers & 加速
pip install -U transformers accelerate
```

> 说明：
>
> * 你之前遇到的 `torch.compiler`、CVE 强制升级等问题，都可通过 **PyTorch ≥ 2.6** 避免。
> * 使用 `device_map="auto"` 需要 `accelerate`（你已安装即可）。

---

## 3. 模型权重获取（两种方式）

### 方式 A｜在线拉取（有网络/镜像时）

首次运行会自动下载权重到 HF 缓存（默认 `~/.cache/huggingface/hub`），但需要配置镜像，不用挂代理也可以下载，并且第一次下载好之后后续运行需要关掉镜像，下面我给出命令，只在linux 下测试过，其他平台请自行测试。

```bash
# 任选其一，第一个就好用，可以直接下载（这个是配置镜像）
export HF_ENDPOINT=https://hf-mirror.com
# 或者
# export HF_ENDPOINT=https://huggingface.co
```

```bash
# 后续下载好之后要用下面命令取消镜像，不然会一直连接huggingface检查，而且还连接失败。
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# 如之前设置过镜像，建议取消
unset HF_ENDPOINT
```

### 看到这里就可以加入自己的文档直接运行代码了，后续内容是开发过程遇到的一些问题，可以先不管
* 向量库构建（首次运行会自动构建向量库，后续运行会自动更新向量库）：
```bash
# 先加入自己的txt，doc，pdf都可以，pdf可能需要安装pdf插件，在requirementations.txt中添加pdf插件，然后运行下面指令构建向量
CUDA_VISIBLE_DEVICES=5 python rag_rs.py --rebuild
```
* 单次问答（便捷）：

```bash
CUDA_VISIBLE_DEVICES=5 python rag_rs.py --ask "Sentinel-2 哪些波段是10m分辨率？"
```

---
后续可以先不看
---

你也可以提前下载：

```bash
# huggingface_hub 方式
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='bge-m3')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='qwen2.5-7b-instruct')"
```

### 方式 B｜离线加载（经量不要，很浪费时间，下载慢传输也慢）

1. 在有网的机器（或本地 Windows 浏览器）下载好 **整个仓库**（BGE-M3、Qwen2.5-7B-Instruct）。
2. 上传到服务器的任意可读写路径，例如：

```
/gpfs-flash/hulab/likai/model_repo/bge-m3/
/gpfs-flash/hulab/likai/model_repo/qwen2.5-7b-instruct/
```

3. 在 `rag_rs.py` 的 `Config` 中改成本地路径（见下一节）。

> 你也可以直接使用 **HF 缓存里的 snapshot 目录**：
> 缓存根见 `python -c "from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)"`
> 常见形如：`$HF_HOME/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<revision>/`

---

## 4. 配置代码（是否离线由你决定）

### 在线加载（保持你当前写法）

```python
@dataclass
class Config:
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
```

### 完全离线加载（不再联网）

把上面两项改成你的**本地绝对路径**，并在加载时加入 `local_files_only=True`：

```python
# 1) Config 改为本地路径
@dataclass
class Config:
    EMBEDDING_MODEL_NAME: str = "/gpfs-flash/hulab/likai/model_repo/bge-m3"
    LLM_MODEL_NAME: str = "/gpfs-flash/hulab/likai/model_repo/qwen2.5-7b-instruct"
```

```python
# 2) build_embeddings 中（你已用 HuggingFaceEmbeddings）
def build_embeddings(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True}
    )
```

```python
# 3) build_llm 中强制离线
tokenizer = AutoTokenizer.from_pretrained(
    cfg.LLM_MODEL_NAME, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    cfg.LLM_MODEL_NAME,
    device_map="auto",        # 用 auto 需要 accelerate；不用就删掉并 model.to("cuda")
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True
)
```

> 可选：运行前设置离线变量，彻底避免网络探测
>
> ```bash
> export TRANSFORMERS_OFFLINE=1
> export HF_HUB_OFFLINE=1
> unset HF_ENDPOINT
> ```

---

## 5. 放知识文档 & 构建向量库

1. 把你的遥感手册/论文/规程放入 `documents/`（支持 TXT、PDF、DOCX）。
   建议优先 TXT（更快更稳），长 PDF 可提要点为 TXT。

2. 首次或文档更新后重建向量库：

```bash
CUDA_VISIBLE_DEVICES=5 python rag_rs.py --rebuild
```

> 备注：Chroma 0.4+ 自动持久化；日志有 deprecation 提示可忽略或删掉 `vs.persist()`。

---

## 6. 运行与测试

* 直接跑（不重建向量库）：

```bash
CUDA_VISIBLE_DEVICES=5 python rag_rs.py
```

* 单次问答（便捷）：

```bash
CUDA_VISIBLE_DEVICES=5 python rag_rs.py --ask "Sentinel-2 哪些波段是10m分辨率？"
```

* 默认 demo 问题已在代码里（遥感相关）；你也可替换。

---

## 7. Windows 终端临时代理（你已用过）

```powershell
# 当前会话有效
set HTTPS_PROXY=http://127.0.0.1:7890
set HTTP_PROXY=http://127.0.0.1:7890
set ALL_PROXY=socks5://127.0.0.1:7890

# 取消
set HTTPS_PROXY=
set HTTP_PROXY=
set ALL_PROXY=
```

---

## 8. 显存与性能建议（3090 24GB）

* 这份代码是 **FP16**：Qwen 7B ≈ 15–17GB；+ BGE/向量库/缓存，总占用 ≈ 18–20GB。
* 控制生成长度：`MAX_NEW_TOKENS=512` 已较稳；如 OOM 降到 256。
* 如果不想装 `accelerate`：删掉 `device_map="auto"`，并手动 `model.to("cuda")`，同时在 `pipeline(..., device=0)` 指定 GPU。

---

## 9. 常见报错 & 快速修复

| 报错/现象                                                         | 原因                                | 解决                                                                                            |
| ------------------------------------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: langchain_community`                    | 版本不匹配/未安装                         | `pip install -U langchain langchain-community langchain-huggingface langchain-text-splitters` |
| `AttributeError: torch has no attribute compiler`             | torch 版本太旧                        | 升级 **torch ≥ 2.1**（你后来已升到 2.6）                                                                |
| `ValueError: upgrade torch to at least v2.6` (CVE-2025-32434) | transformers 禁止旧版 torch 加载 `.bin` | **torch ≥ 2.6** 或使用 `.safetensors`                                                            |
| `requires accelerate`                                         | 使用 `device_map="auto"`            | `pip install -U accelerate` 或去掉 `device_map` 并 `model.to("cuda")`                             |
| 反复请求 `hf-mirror.com`                                          | 仍在线模式/设置了镜像                       | 改本地路径 + `local_files_only=True`，或 `TRANSFORMERS_OFFLINE=1; unset HF_ENDPOINT`                 |
| `Chroma persist 警告`                                           | 0.4+ 自动持久化                        | 无需处理，或删掉 `vs.persist()`                                                                       |
| 运行时 OOM                                                       | 上下文过长/并发过多                        | 降低 `MAX_NEW_TOKENS`、减少并发、关闭其他占显存进程                                                            |

---

## 10. 日常使用流

1. 往 `documents/` 里加/删文档
2. `CUDA_VISIBLE_DEVICES=X python rag_rs.py --rebuild`
3. 直接问答：`--ask "你的问题"` 或运行后输入问题
4. 若要完全离线：改本地路径 + `local_files_only=True` +（可选）OFFLINE 环境变量

---

## 11. 可选优化（进阶）

* **检索多样性**：改 `retriever` 为 `search_type="mmr"`，`k=6`，`lambda_mult=0.5`。
* **减少幻觉**：在系统 prompt 中强调“仅依据上下文，无法确定时说明‘无依据’”。
* **文档分块**：长文档用更大的 `CHUNK_SIZE`（如 1000–1200）+ 适度重叠。
* **固定模型版本**：`from_pretrained(..., revision="具体commit")`，防止仓库更新导致再次联网校验。

---

## 12. 一键“完全离线”运行示例

```bash
# (可选) 迁移/固定 HF 缓存根到可写目录
export HF_HOME=/gpfs-flash/hulab/likai/.hf_cache
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME

# 强制离线
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HF_ENDPOINT

# 运行（如首次构建）
CUDA_VISIBLE_DEVICES=5 python rag_rs.py --rebuild
```

---

需要我把你的 `rag_rs.py` **直接改成“本地路径 + 完全离线版”** 并贴出来吗？
把你两个本地模型的**最终绝对路径**（BGE 与 Qwen）发我，我就把相应的几行替换好给你。
