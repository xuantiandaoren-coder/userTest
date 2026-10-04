"""RAG 模块：文件向量化入库与知识库检索。

- 入库：``ingest_file``（app/rag/core.py，解析 -> 分块 -> 向量化 -> 写 Qdrant + 写分块原文）
- 检索：``search_similar_chunks``（app/rag/retriever.py，问题向量化 -> Qdrant 近邻 -> 回 MySQL 取原文）
- 分块原文读写：``app/rag/chunk_store.py``；扫描件 OCR 兜底：``app/rag/ocr.py``
"""

from app.rag.core import ingest_file
from app.rag.retriever import search_similar_chunks

__all__ = ["ingest_file", "search_similar_chunks"]
