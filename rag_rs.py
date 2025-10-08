# rag_rs.py
# -*- coding: utf-8 -*-
import os, time, argparse
from dataclasses import dataclass

import torch
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline
)

# ==================================================
# Config
# ==================================================
@dataclass
class Config:
    # 文档与向量库
    DOCUMENTS_DIR: str = "documents"
    VECTOR_DB_DIR: str = "vector_db_qwen_bge_m3"

    # 模型
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    # EMBEDDING_MODEL_NAME: str = "/gpfs-flash/hulab/likai/zxc/ly/rag/model_repo/bge-m3"
    # LLM_MODEL_NAME: str = "/gpfs-flash/hulab/likai/zxc/ly/rag/model_repo/Qwen2.5-7B-Instruct"
    # 分块策略（中文友好）
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 4

    # 生成参数
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.2
    REPETITION_PENALTY: float = 1.05


def build_embeddings(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # BGE-M3 推荐加查询指令；HuggingFaceEmbeddings 用 encode_kwargs 传递
    return HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL_NAME,   # "BAAI/bge-m3"
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )



# ==================================================
# LLM (Qwen) - FP16 模式
# ==================================================
def build_llm(cfg: Config):
    print(f"[INFO] 加载 Qwen 模型（FP16 模式）: {cfg.LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.LLM_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,   # 半精度 FP16，适配 RTX 3090
        trust_remote_code=True
    )

    # Qwen 对话格式包装
    def format_prompt(user_prompt: str) -> str:
        return (
            "<|im_start|>system\n你是一个可靠的中文遥感专家助手，严格依据提供的上下文回答，并在必要时给出出处。<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.MAX_NEW_TOKENS,
        temperature=cfg.TEMPERATURE,
        repetition_penalty=cfg.REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )

    class WrappedLLM(HuggingFacePipeline):
        def _call(self, prompt: str, stop=None):
            return super()._call(format_prompt(prompt), stop=stop)

    return WrappedLLM(pipeline=gen_pipe)


# ==================================================
# 文档加载
# ==================================================
def iter_load_documents(doc_dir: str):
    docs = []
    txt_loader = DirectoryLoader(doc_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs += txt_loader.load()

    for root, _, files in os.walk(doc_dir):
        for f in files:
            path = os.path.join(root, f)
            if f.lower().endswith(".pdf"):
                docs += PyPDFLoader(path).load()
            if f.lower().endswith(".docx"):
                docs += Docx2txtLoader(path).load()
    return docs


# ==================================================
# 构建 / 加载向量库
# ==================================================
def build_or_load_vector_db(cfg: Config, embeddings, force_rebuild=False):
    if (not force_rebuild) and os.path.isdir(cfg.VECTOR_DB_DIR) and os.listdir(cfg.VECTOR_DB_DIR):
        print(f"[OK] 加载已存在的向量库: {cfg.VECTOR_DB_DIR}")
        return Chroma(persist_directory=cfg.VECTOR_DB_DIR, embedding_function=embeddings)

    print("[*] 重新构建向量库...")
    raw_docs = iter_load_documents(cfg.DOCUMENTS_DIR)
    if not raw_docs:
        raise RuntimeError(f"请把遥感 PDF/TXT 放到 {cfg.DOCUMENTS_DIR}/ 后再运行。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", "、", " ", ""]
    )
    texts = splitter.split_documents(raw_docs)
    print(f"[*] 已分块：{len(texts)} 个 chunk")

    vs = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=cfg.VECTOR_DB_DIR
    )
    vs.persist()
    print(f"[OK] 向量库已持久化：{cfg.VECTOR_DB_DIR}")
    return vs


# ==================================================
# 问答链
# ==================================================
def build_qa_chain(llm, vector_db, top_k: int):
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


# ==================================================
# 主程序
# ==================================================
def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量库")
    parser.add_argument("--ask", type=str, default=None, help="直接提问并退出")
    args = parser.parse_args()

    cfg = Config()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(cfg.DOCUMENTS_DIR, exist_ok=True)

    print("[1/4] 加载嵌入模型 BGE-M3 ...")
    embeddings = build_embeddings(cfg)

    print("[2/4] 加载 Qwen LLM (FP16) ...")
    llm = build_llm(cfg)

    print("[3/4] 加载/构建向量库 ...")
    vs = build_or_load_vector_db(cfg, embeddings, force_rebuild=args.rebuild)

    print("[4/4] 构建检索问答链 ...")
    qa = build_qa_chain(llm, vs, cfg.TOP_K)

    def ask(q):
        t0 = time.time()
        out = qa({"query": q})
        dt = time.time() - t0
        print("\n【问题】", q)
        print("【回答】", out["result"].strip())
        print("【耗时】%.2fs" % dt)
        print("【溯源片段】")
        for i, d in enumerate(out["source_documents"], 1):
            src = d.metadata.get("source", "unknown")
            print(f"  [{i}] {src}")
        print("-" * 60)

    if args.ask:
        ask(args.ask)
        return

    demo_questions = [
        "Sentinel-2 的 13 个波段及其空间分辨率是什么？各用于什么场景？",
        "Landsat Collection 2 Level-2 的表面反射率产品有什么改进？",
        "QA60 云掩膜是什么？为什么很多人更推荐 Fmask 或 s2cloudless？",
        "常见的植被指数 NDVI/EVI 的定义和适用场景是什么？",
    ]
    for q in demo_questions:
        ask(q)

if __name__ == "__main__":
    main()



# CUDA_VISIBLE_DEVICES=5 python rag_rs.py --rebuild
# CUDA_VISIBLE_DEVICES=5 python rag_rs.py --ask "Sentinel-2 哪些波段是10m分辨率？"

