"""RAG 模块：文件向量化入库。

对外只暴露 `ingest_file`，实现全部收敛在 app/rag/core.py 单文件内。
"""

from app.rag.core import ingest_file

__all__ = ["ingest_file"]
