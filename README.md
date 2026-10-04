# FastAPI 用户信息管理接口

基于 **FastAPI + SQLAlchemy 2.0 + MySQL + Alembic** 的用户增删改查接口示例，
代码按四层组织：路由层、校验层、业务层、数据库层。

表结构变更由 Alembic 管理，改完模型后执行一条命令即可同步到数据库。

## 技术要点

- 依赖与虚拟环境由 `uv` 管理（Python 3.12）
- ORM 使用 SQLAlchemy 2.0 声明式模型（`Mapped` / `mapped_column`），驱动为 PyMySQL
- 表结构变更由 Alembic 管理，支持 autogenerate 一键同步
- 配置分层：`APP_ENV` 区分 dev / prod，按「环境变量 > `.env.<环境>` > `.env`」加载
- 敏感信息（连接串、JWT 密钥）只走配置层，用 `SecretStr` 包裹，不写死、不落日志
- JWT 鉴权：PyJWT 签发 access / refresh 令牌，受保护接口用 Bearer 校验
- 日志体系：控制台 + 按大小轮转的文件日志，分级记录、每请求一条访问日志
- 文件上传（元数据与文件解耦）：MySQL `resources` 只存元数据，原文件存 **SeaweedFS** 对象存储；
  按内容识别图片 / 文档 / 音频，MD5 + 用户唯一索引去重，`upload_purpose=1` 的图片写用户头像
- 请求/响应校验使用 Pydantic v2，配合 FastAPI `response_model`
- 密码使用 `passlib` + `bcrypt` 加盐哈希，数据库只存哈希，不保留明文
- 对外响应仅返回 `id`、`username`，永不泄露密码字段
- AI 聊天流式响应：`POST /sessions/{session_id}/stream-chat` 用 SSE 逐字下发，
  按「模型层 / 提示词层 / 记忆层」三层解耦（见下文「AI 聊天流式响应」）
- 提示词版本管理：`prompt_templates` 表按 `(agent_name, scene)` 分组多版本共存、单版本生效，
  支持一键回滚，并用 Redis Hash 缓存生效模板（见下文「提示词模板版本管理」）

## 项目结构

```text
.
├── app
│   ├── main.py                 # FastAPI 应用入口（关闭时释放连接池）
│   ├── core
│   │   ├── config.py           # 配置层：dev/prod 分层、敏感信息用 SecretStr
│   │   ├── exceptions.py       # 业务 / 系统两类自定义异常
│   │   ├── handlers.py         # 全局异常处理器（标准响应 + 分级日志）
│   │   ├── logging.py          # 日志体系：控制台 + 轮转文件，幂等配置
│   │   ├── middleware.py       # 访问日志中间件（每请求一条）
│   │   ├── security.py         # 密码强度校验（黑名单/字母数字/不含用户名）
│   │   ├── tokens.py           # JWT 签发与校验（access / refresh）
│   │   ├── storage.py          # 类型识别 + 本地落盘（旧逻辑教学用）+ 上传内容读取
│   │   ├── seaweedfs.py        # SeaweedFS 对象存储客户端（S3 网关，put / delete 对象）
│   │   ├── extract.py          # 文件内容提取（storage_scene=2 只取内容不存原文件）
│   │   ├── scheduler.py        # 轻量定时器：每天固定时刻跑一次任务
│   │   ├── redis_client.py     # 提示词模板缓存用的 Redis 客户端（惰性连接 + 失败降级）
│   │   ├── sse.py              # SSE 帧序列化（text/event-stream）
│   │   └── rate_limit.py       # 固定窗口限流（IP 维度）
│   ├── llm
│   │   └── llm.py              # 模型层：provider -> 模型实例（DeepSeek/OpenAI/通义/…）
│   ├── memory
│   │   └── memory.py           # 记忆层：历史轮次构建 + 跨会话检索增强
│   ├── prompts
│   │   ├── prompt_template_manager.py  # 版本计算 / Redis 缓存 / 公共+私有拼接 / 一键回滚
│   │   ├── injector.py         # 变量注入：验证层 -> 转换层 -> 填充层
│   │   └── prompt_layer.py     # 提示词层：Prompt 模板 + LCEL 链路组装
│   ├── clients.py              # 调用方自动刷新客户端（httpx，服务端不依赖）
│   ├── api                     # 路由层
│   │   ├── router.py           #   路由汇总
│   │   ├── deps.py             #   公共依赖：仓储 + 业务服务 + Bearer 鉴权
│   │   ├── auth.py             #   /auth/register 注册、/auth/login 登录、/auth/refresh 刷新
│   │   ├── files.py            #   POST /files/upload、/upload/file 需认证的通用文件上传
│   │   ├── health.py           #   GET /health 健康检查
│   │   ├── interviews.py       #   GET /interviews/{id} 面试详情
│   │   ├── prompt.py           #   /prompt 模板版本管理（查看 / 新建 / 回滚 / 智能体配置）
│   │   ├── sessions.py         #   会话增删改查 + 会话消息分页 + SSE 流式聊天
│   │   └── users.py            #   用户增删改查接口
│   ├── schemas                 # 校验层（Pydantic v2 请求/响应模型）
│   │   ├── auth.py             #   注册请求模型（附加密码强度校验）
│   │   ├── chat.py             #   会话 / 消息（附件段）/ 面试模型
│   │   ├── prompt.py           #   提示词模板 / 回滚 / 智能体配置模型
│   │   ├── common.py           #   统一错误响应模型 + 通用分页壳 Page[T]
│   │   ├── file.py             #   文件上传响应模型
│   │   ├── health.py           #   健康检查响应模型
│   │   └── user.py
│   ├── services                # 业务层
│   │   ├── chat_service.py     #   会话 / 消息 / 面试业务逻辑 + 级联删除
│   │   ├── stream_chat_service.py  # 流式聊天：模板 + 变量 + 记忆 + 模型链路 -> SSE
│   │   ├── upload_service.py   #   旧本地落盘实现（教学保留，运行链路不经过）
│   │   ├── upload_service_seaweedfs.py  # 运行链路：SeaweedFS 存原文件 + MySQL 存元数据
│   │   ├── rag_service.py      #   上传后 RAG 向量化入库（best-effort，失败不影响上传）
│   │   ├── resource_cleanup.py #   每天 03:00 清理过期资源（先删对象，再删元数据）
│   │   └── user_service.py     #   业务逻辑 + 密码哈希
│   ├── rag                     # RAG 入库实现
│   │   ├── core.py             #   解析 -> 清洗 -> 分类 -> 分块 -> 向量化 -> 写 Qdrant + 写分块原文
│   │   ├── chunk_store.py      #   分块原文写 MySQL knowledge_chunks（与 Qdrant 用 UUID 关联）
│   │   ├── retriever.py        #   检索：问题向量化 -> Qdrant 近邻（按 user_id 过滤）-> 回 MySQL 取原文
│   │   ├── dialogue.py         #   对话链路：意图判断 -> 资料注入 -> 来源组装与历史回显
│   │   └── ocr.py              #   扫描件 OCR 兜底：渲染 PNG -> 视觉 OCR 模型 -> 按行拼接
│   └── db                      # 数据库层
│       ├── base.py             #   SQLAlchemy 声明式基类（含约束命名规范）
│       ├── chat_message_repository.py  # 消息表数据访问（分页 / 附件段解析）
│       ├── interview_repository.py     # 面试表数据访问（id+user 查询）
│       ├── models.py           #   ORM 模型：用户 / 会话 / 消息 / 面试 / 资源 / 知识库分块 / 提示词 / 画像
│       ├── prompt_template_repository.py  # prompt_templates 表数据访问（分组 / 版本 / 生效切换）
│       ├── resource_repository.py  # resources 表数据访问（去重预查 / 过期扫描）
│       ├── session_repository.py   # 会话表数据访问（分页 / 归属过滤）
│       ├── user_profile_repository.py  # user_profiles 表数据访问（变量注入的数据来源）
│       ├── user_repository.py  #   用户表数据访问（SQL 收敛在这里）
│       └── session.py          #   引擎 / 会话 / get_db 依赖
├── alembic
│   ├── env.py                  # 连接串取自应用配置，元数据取自 ORM 模型
│   └── versions                # 迁移脚本
├── scripts
│   └── db_sync.py              # 一键同步表结构
├── tests                       # pytest 接口测试（用 SQLite 替换 MySQL）
├── logs                        # 运行时日志目录（已忽略，不入库）
├── storage                     # 旧本地落盘实现的目录（教学用；运行链路写 SeaweedFS）
├── alembic.ini
├── .env.example                # 基础配置模板（入库）
├── .env.dev.example            # 开发覆盖项模板（入库）
├── .env.prod.example           # 生产覆盖项模板（入库）
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## 数据建模（`user` 表）

| 列名 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `id` | `INT` | 主键、自增 | 主键 |
| `userName` | `VARCHAR(20)` | 唯一索引 `uk_user_userName`、非空 | 用户名，API 字段名为 `username` |
| `password` | `VARCHAR(255)` | 非空 | bcrypt 密码哈希（非明文） |
| `create_time` | `DATETIME` | 非空、`DEFAULT CURRENT_TIMESTAMP` | 由数据库自动填充当前时间 |
| `avatar` | `VARCHAR(255)` | 可空 | 头像对象键 `<user_id>/<md5>.<ext>`（SeaweedFS），`upload_purpose=1` 的图片写入；接口返回时同时给出可访问的 `avatar_url` |

模型定义见 `app/db/models.py`，对应迁移脚本见 `alembic/versions/20260912_0001_create_user_table.py`。

## 数据建模（`resources` 资源元数据表）

原文件放 SeaweedFS，本表只存元数据，两者解耦；`UNIQUE(file_hash, user_id)` 是用户级去重的核心。
模型定义见 `app/db/models.py` 的 `Resource`，迁移脚本 `alembic/versions/20260915_0004_create_resources.py`。
`doc_category` 列由 `alembic/versions/20260930_0007_add_resource_doc_category.py` 追加。

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `id` | `BIGINT` | 主键、自增 |
| `resource_type` | `TINYINT` | 0=文件，1=图片，2=音频 |
| `doc_category` | `VARCHAR(32)` | 文档分类 `resume` / `study_material` / `general`，可空；**仅文件类型（`resource_type=0`）有意义**，图片 / 音频恒为 NULL |
| `storage_scene` | `TINYINT` | 0=长过期（1 个月），1=短过期（2 小时），2=只提取内容不存原文件 |
| `upload_purpose` | `TINYINT` | 0=普通资源，1=用户头像 |
| `file_name` | `VARCHAR(255)` | 用户上传的原始文件名（只取 basename） |
| `file_hash` | `VARCHAR(64)` | 文件内容 MD5，去重核心字段 |
| `storage_path` | `VARCHAR(512)` | SeaweedFS 对象键（`storage_scene=2` 不落表） |
| `user_id` | `BIGINT` | 上传用户 ID |
| `expire_time` | `DATETIME` | 过期时间，到期后被每日清理任务删除 |
| `create_time` | `DATETIME` | 创建时间，由数据库填充 |

### 过期清理

应用启动时拉起常驻协程，**每天 03:00** 扫描 `expire_time <= now` 的资源：
先删 SeaweedFS 对象，成功后再删元数据（删对象失败则保留元数据、次日重试，避免产生无主对象）。
清理逻辑见 `app/services/resource_cleanup.py`，调度器见 `app/core/scheduler.py`。

## 数据建模（`knowledge_chunks` 知识库分块原文表）

分块数据**分两份存**：向量在 Qdrant，原文在本表，两边用入库时生成的同一个 UUID 关联
（Qdrant point id == `knowledge_chunks.vector_id`）。
模型定义见 `app/db/models.py` 的 `KnowledgeChunk`，迁移脚本
`alembic/versions/20261004_0008_create_knowledge_chunks.py`，写入实现见 `app/rag/chunk_store.py`。

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `id` | `BIGINT` | 主键、自增 |
| `resource_id` | `BIGINT` | 关联 `resources.id`（外键 `ON DELETE CASCADE`，资源过期被清理时本表跟随删除） |
| `vector_id` | `VARCHAR(36)` | Qdrant point id（入库时生成的 UUID），唯一索引 `uk_knowledge_chunks_vector_id` |
| `chunk_index` | `INT` | 文件内分块序号，从 0 开始 |
| `char_count` | `INT` | 分块字符数 |
| `text` | `MEDIUMTEXT` | 分块原文 |
| `created_at` | `BIGINT` | 入库时间（Unix 秒，数据库填充） |

检索链路：先用查询向量在 Qdrant 召回 point id（顺带按 `user_id` / `doc_category` 过滤），
再按 `vector_id` 回本表取原文——Qdrant 侧因此只存向量与过滤字段，向量库体积与 payload 写入都更小。

### 检索（`app/rag/retriever.py`）

对外只暴露 `search_similar_chunks(query, *, user_id, db, top_k=3)`，三步：

1. 用与入库同一个模型把问题向量化（`core.embed_texts`，维度 / Key 完全共用）
2. 在 Qdrant 里按余弦相似度取最近 `top_k` 个点，**必带 `user_id` 过滤**：
   所有用户共用一个 collection，不过滤会召回别人的资料
3. 拿命中的 point id（== `knowledge_chunks.vector_id`）回 MySQL 批量取原文
   （`chunk_store.fetch_chunks_by_vector_ids`，**同样带 `user_id` 校验归属**，作为第二道隔离），
   再按相似度从高到低组装

结果整形：**相似度低于 `MIN_SCORE`（0.4）的命中丢弃**（Qdrant 侧 `score_threshold` 先滤，
返回前再兜一次）；**单条原文超过 `MAX_TEXT_CHARS`（500 字）截断**（只影响返回值，
库里原文不动）；`top_k` 由调用方控制，默认 `DEFAULT_TOP_K`（3）。

```python
from app.rag import search_similar_chunks

hits = search_similar_chunks("HashMap 的扩容因子是多少", user_id=7, db=session, top_k=3)
context = "\n\n".join(hit.text for hit in hits)   # 直接拼给模型做回答
for hit in hits:
    print(hit.score, hit.file_name, hit.chunk_index)
```

返回 `ChunkHit` 列表（`vector_id` / `text` / `score` / `file_name` / `doc_category` / `chunk_index`，
可 `as_dict()` 序列化）。边界行为：空问题、`top_k <= 0` 直接返回空列表且不调模型；
collection 还没建（没上传过文档）返回空列表；向量库有、MySQL 没有的悬空点跳过并告警；
Qdrant 异常抛 `RETRIEVAL_FAILED`。

### RAG 对话链路（`app/rag/dialogue.py`）

流式聊天按需检索自己的资料并注入本轮上下文，四步都在 `app/rag/dialogue.py`：

1. **意图判断** `should_retrieve(request_text)`：空输入、极短输入（去标点后 < 4 字）、
   确认语（好的 / 嗯 / ok）、礼貌语（你好 / 谢谢）、推进语（继续 / 下一题）都不检索——
   这类消息花一次向量化 + 一次向量检索没有收益。短组合（"好的，继续"）同样跳过
2. **检索** `retrieve_knowledge(query, user_id=…, db=…, top_k=3)`：按 `user_id` 隔离调
   `search_similar_chunks`；检索失败只记日志，本轮按无资料回答（不影响对话本身）
3. **注入** `format_retrieved_chunks` 把命中片段排成 `【参考资料】` 文本，
   `compose_question` 把它拼到**本轮用户问题前**；`build_rag_system_prompt` 只在原
   system 提示词后补充 `【RAG 回答规则】`（要求引用编号、不许编造），不改写模板正文。
   注意：注入只进本轮提示词，落库与历史里的 `request_text` 仍是**原始提问**
4. **来源** `build_sources` 给完整来源（`chunk_id` / `resource_id` / `chunk_index` /
   `file_name` / `score` / `text`），SSE `done.sources` 下发；
   `build_references` 给落库引用（只有 `chunk_id` + `score`），写进
   `chat_messages.reference_sources`——**知识块全文不在消息表重复保存**

```jsonc
// SSE done 里的 sources
{"sources": [{"chunk_id": "10cab2fd-…", "resource_id": 13, "chunk_index": 0,
              "file_name": "java.pdf", "score": 0.7182, "text": "HashMap 底层是数组加链表…"}]}
```

**历史回显**：`GET /sessions/{id}/messages` 的每条消息带 `sources`（`refreshed` 场景同源）。
实现是 `resolve_reference_sources` 按 `reference_sources` 里的 `chunk_id` 回查
`knowledge_chunks` + `resources` 拼回完整来源，并同样按 `user_id` 校验归属；
引用的知识块已被删除（资源过期清理 / 重新上传覆盖）时，`text` 返回
**「该参考片段已经删除」**、出处字段为 `null`，`score` 仍回放落库值。列表接口按页只查一次
（`load_source_index`），回查失败降级为空来源，不让历史列表整体报错。

## 数据建模（会话 / 消息 / 面试）

三张业务表同在 `app/db/models.py`，对应迁移脚本 `alembic/versions/20260914_0003_create_chat_tables.py`
（附件段与 `interviews.user_id` 由 `alembic/versions/20260915_0005_chat_segments_and_interview_user.py` 追加）。
时间戳统一用 `BIGINT` 存 Unix 秒，默认值由数据库写入（MySQL `DEFAULT (UNIX_TIMESTAMP())`，
测试用的 SQLite 退化为 `strftime('%s','now')`），避免多台应用服务器时钟不一致。

### `sessions`（会话）

| 列名 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `id` | `INT` | 主键、自增 | 主键 |
| `user_id` | `INT` | 外键 `user.id`、索引 `ix_sessions_user_id`、非空 | 所属用户 |
| `session_model` | `INT` | 非空 | 0=学习，1=面试，2=笔记 |
| `title` | `VARCHAR(255)` | 非空 | 会话标题 |
| `created_at` | `BIGINT` | 非空、`DEFAULT (UNIX_TIMESTAMP())` | 会话创建时间（Unix 秒） |

### `chat_messages`（消息）

| 列名 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `id` | `INT` | 主键、自增 | 主键 |
| `user_id` | `INT` | 外键 `user.id`、非空 | 所属用户 |
| `session_id` | `INT` | 外键 `sessions.id`、非空 | 所属会话 |
| `select_model` | `INT` | 非空 | 0=默认，1=知识精讲，2=刷题，3=简历优化，4=模拟面试，5=面试复盘 |
| `request_id` | `VARCHAR(64)` | 索引 `ix_chat_messages_request_id`、非空 | 请求唯一标识（幂等 / 链路追踪） |
| `request_text` | `MEDIUMTEXT` | 非空 | 提问文本 |
| `response_text` | `MEDIUMTEXT` | 非空 | 回答文本 |
| `request_segments` | `JSON` | 可空 | 提问附件段，只存 `file`/`image`/`audio`，元素如 `{"type":"image","resource_id":13}` |
| `response_segments` | `JSON` | 可空 | 回复附件段，同上；AI 回复也支持只带附件 |
| `file_extracted_text` | `MEDIUMTEXT` | 可空 | 从文件提取的完整文本（对话上下文用） |
| `reference_sources` | `JSON` | 可空 | 本轮回答引用的知识片段，**只存引用不存全文**，元素如 `{"chunk_id":"<knowledge_chunks.vector_id>","score":0.78}` |
| `created_at` | `BIGINT` | 非空、`DEFAULT (UNIX_TIMESTAMP())` | 创建时间（Unix 秒） |

复合索引 `ix_chat_messages_session_id_created_at`（`session_id`, `created_at`）：按会话拉取消息列表。

`reference_sources` 由 `alembic/versions/20261004_0009_add_chat_message_reference_sources.py` 追加：
正文（知识块全文）只在 `knowledge_chunks` 存一份，消息表只记 chunk_id 与相似度，
历史消息接口据此回查原文重新拼出 `sources`（见下文「RAG 对话链路」）。

### `interviews`（面试记录）

| 列名 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `id` | `INT` | 主键、自增 | 主键 |
| `session_id` | `INT` | 外键 `sessions.id`、非空 | 所属会话（同一面试场景） |
| `user_id` | `INT` | 外键 `user.id`、索引 `ix_interviews_user_id`、非空 | 所属用户；面试详情按 `interview_id` + `user_id` 查询，避免越权 |
| `message_id` | `INT` | 外键 `chat_messages.id`、唯一 `uq_interviews_message_id`、非空 | 开启本次模拟面试的入口消息 |
| `qa_object` | `JSON` | 非空 | 一问一答对象列表 |
| `interview_duration` | `INT` | 非空、`DEFAULT 0` | 累计面试时长（秒） |
| `status` | `INT` | 非空、`DEFAULT 0`、索引 `ix_interviews_status` | 0=进行中，1=已结束，2=异常终止 |
| `created_at` | `BIGINT` | 非空、`DEFAULT (UNIX_TIMESTAMP())` | 面试开始时间（Unix 秒） |
| `updated_at` | `BIGINT` | 非空、`DEFAULT (UNIX_TIMESTAMP())` | 面试更新时间（Unix 秒），ORM 更新时自动刷新 |

复合索引 `ix_interviews_session_id_message_id`（`session_id`, `message_id`）。

`qa_object` 元素结构（`created_at` 为 Unix 秒，与表字段对齐）：

```json
[
    {"id": "uuid", "question": "...", "answer": "...", "created_at": 1717171717}
]
```

## 数据建模（`prompt_templates` 提示词模板表）

一个 `(agent_name, scene)` 就是一组模板，组内多版本共存、只有一版生效，「一键回滚」就是切换生效版本。
模型定义见 `app/db/models.py` 的 `PromptTemplate`，迁移脚本
`alembic/versions/20260915_0006_create_prompt_and_profile.py`。

| 列名 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `id` | `INT` | 主键、自增 | 主键（回滚接口传的就是它） |
| `agent_name` | `VARCHAR(64)` | 可空 | 智能体名；**公共模板为 NULL** |
| `scene` | `VARCHAR(64)` | 非空 | 场景：`workflow` / `resume` / `quiz` / `interview` / `evaluation` / `study` / `note` |
| `template_type` | `TINYINT` | 非空、`DEFAULT 1` | 1=私有（`agent_name` 有值），2=公共（`agent_name` 为空） |
| `template_content` | `MEDIUMTEXT` | 非空 | 模板正文，用 `{变量}` 占位 |
| `variables` | `VARCHAR(512)` | 可空 | 变量名列表（JSON 数组字符串，如 `["target_job"]`） |
| `version` | `INT` | 非空、`DEFAULT 1` | 版本号：**同一 `(agent_name, scene)` 内自增，从 1 开始** |
| `is_active` | `TINYINT` | 非空、`DEFAULT 0` | 是否生效；**同一组最多一条为 1** |
| `description` | `VARCHAR(255)` | 可空 | 版本说明（这一版改了什么） |
| `created_at` | `BIGINT` | 非空、`DEFAULT (UNIX_TIMESTAMP())` | 创建时间（Unix 秒） |

索引 `ix_prompt_templates_agent_scene`（`agent_name`, `scene`, `is_active`）。

关键规则（由服务层 `app/prompts/prompt_template_manager.py` 保证）：

1. **公共模板**：`agent_name IS NULL` + `template_type=2`；**私有模板**：`agent_name` 有值 + `template_type=1`（两者不一致直接 422）
2. **版本独立**：`(agent_name, scene)` 变更即换一组，版本号互不影响（`difficulty_learner:study` 到 v3 不影响 `quiz_generate_workflow:quiz` 仍是 v1）
3. **仅一条生效**：切版本时在同一事务里先 `is_active=0` 整组、再把目标版本置 1；不依赖唯一索引——
   MySQL / SQLite 的唯一索引里 `NULL` 互不相等，公共模板（`agent_name=NULL`）无法靠唯一索引约束

## 数据建模（`user_profiles` 用户背景画像表）

提示词变量注入的数据来源（一人一行），模型见 `app/db/models.py` 的 `UserProfile`。

| 列名 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `id` | `INT` | 主键、自增 | 主键 |
| `user_id` | `INT` | 外键 `user.id`、唯一索引 `uk_user_profiles_user_id`、非空 | 所属用户（一人一行） |
| `target_job` | `VARCHAR(128)` | 可空 | 目标岗位 |
| `years_experience` | `INT` | 可空 | 工作经验（年），注入时转成「3 年」 |
| `target_level` | `VARCHAR(32)` | 可空 | 目标等级，如 `P6` / `高级` |
| `target_skills` | `JSON` | 可空 | 已掌握技能（JSON 数组），注入时转成顿号分隔文本 |
| `weak_topics` | `JSON` | 可空 | 薄弱点（JSON 数组） |
| `created_at` | `BIGINT` | 非空、`DEFAULT (UNIX_TIMESTAMP())` | 创建时间（Unix 秒） |

## 快速开始

```bash
# 1. 安装依赖（自动生成 uv.lock）
uv sync

# 2. 配置：基础配置 + 环境专属覆盖
cp .env.example .env            # 基础配置（含 MYSQL_URL 等敏感信息）
cp .env.dev.example .env.dev    # 开发覆盖项：DEBUG 日志、开启 SQL 打印
#   编辑 .env，把 MYSQL_URL 改成：
#   MYSQL_URL=mysql+pymysql://user:password@127.0.0.1:3306/user_api?charset=utf8mb4

# 3. 在 MySQL 中先建好库（表由迁移创建）
mysql -uroot -p -e "CREATE DATABASE user_api DEFAULT CHARSET utf8mb4;"

# 4. 建表 / 升级到最新表结构
uv run alembic upgrade head

# 5. 启动服务（默认 http://127.0.0.1:8000）
uv run uvicorn app.main:app --reload
```

连接串也可以不写 `.env`，直接用环境变量覆盖：`export MYSQL_URL=...`（环境变量优先级高于 `.env`）。

未配置连接串时服务仍能启动，只在真正访问数据库时提示
`MySQL 连接串未配置：请复制 .env.example 为 .env 并填写 MYSQL_URL`。

### 连接串中的特殊字符

密码含有 `@`、`/`、`#`、`%` 等字符时，必须按 URL 编码写入配置：`@` -> `%40`、`/` -> `%2F`、
`#` -> `%23`、`%` -> `%25`。例如密码 `p@ss/word`：

```bash
MYSQL_URL=mysql+pymysql://app:p%40ss%2Fword@127.0.0.1:3306/user_api?charset=utf8mb4
```

**不需要在代码里手动解码**：SQLAlchemy 解析连接串时会自动还原（`make_url()` 内部对用户名、密码做
`urllib.parse.unquote`），PyMySQL 最终拿到的就是明文密码 `p@ss/word`。反过来，如果传入的密码已经解码，
URL 会在 `@` 处被提前截断，反而连不上。

Alembic 侧也已处理：`alembic.ini` 基于 `ConfigParser`，默认会对 `%` 做插值，编码后的密码会触发
`invalid interpolation syntax`，因此 `alembic/env.py` 改为 raw 读取连接串（未配置时回落到 `settings`），
密码中的 `%` 不会被二次处理。

启动后访问：

- Swagger 接口文档：http://127.0.0.1:8000/docs
- 健康检查（根路径）：http://127.0.0.1:8000/

## 配置分层（dev / prod）

运行环境由 `APP_ENV` 决定（`dev` / `prod`，缺省 `dev`），配置文件按优先级从低到高加载：

| 优先级 | 来源 | 用途 |
| --- | --- | --- |
| 1（最高） | 进程环境变量 | 容器 / CI 注入，如 `MYSQL_URL`、`LOG_LEVEL` |
| 2 | `.env.<APP_ENV>` | 环境专属覆盖，如 `.env.dev`、`.env.prod` |
| 3 | `.env` | 各环境共享的基础配置 |
| 4（最低） | 代码默认值 | 只放非敏感、可公开的项 |

```bash
# 开发：.env + .env.dev
uv run uvicorn app.main:app --reload

# 生产：.env + .env.prod，并显式声明环境
APP_ENV=prod uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

主要配置项（完整清单见 `.env.example`）：

| 变量 | 说明 | dev 默认 | prod 默认 |
| --- | --- | --- | --- |
| `APP_ENV` | 运行环境，非法值直接启动失败 | `dev` | `prod`（必须显式设置） |
| `MYSQL_URL` | MySQL 连接串（敏感） | 占位符，访问数据库时提示未配置 | **必填**，缺失直接启动失败 |
| `DOCS_ENABLED` | 是否暴露 `/docs`、`/redoc`、`/openapi.json` | `true` | `false` |
| `LOG_LEVEL` | 日志级别 | `INFO` | `INFO` |
| `LOG_DIR` / `LOG_FILE` | 日志目录与文件名 | `logs/app.log` | 同左；生产建议改 `/var/log/user-api`（`.env.prod.example` 已给出） |
| `LOG_MAX_BYTES` / `LOG_BACKUP_COUNT` | 轮转大小与保留份数 | 10MB / 7 | 同左 |
| `LOG_ACCESS` | 是否记录访问日志 | `true` | `true` |
| `DB_ECHO` | 是否打印 SQL | `false` | `false` |
| `DB_CONNECT_TIMEOUT` | 建连超时秒数，数据库不可达时快速失败 | `5` | `5` |
| `STORAGE_ROOT` | 旧本地落盘实现的根目录（教学用，运行链路不使用） | `storage` | 同左 |
| `MAX_UPLOAD_BYTES` | 单文件大小上限（字节） | `10485760`（10MB） | 同左 |
| `SEAWEEDFS_ENDPOINT` | SeaweedFS S3 网关地址 | `http://127.0.0.1:8333` | 指向内网网关，如 `http://seaweedfs.internal:8333` |
| `SEAWEEDFS_ACCESS_KEY` / `SEAWEEDFS_SECRET_KEY` | 对象存储凭据（敏感，`SecretStr`） | 空 | 密钥管理服务注入 |
| `SEAWEEDFS_BUCKET` | 默认 bucket | `test` | 建议按环境分桶 |
| `SEAWEEDFS_REGION` | S3 签名区域（SeaweedFS 不校验，占位） | `us-east-1` | 同左 |
| `SEAWEEDFS_PUBLIC_BASE_URL` | 对象对外访问前缀（CDN / 反向代理），留空用 `<endpoint>/<bucket>` | 空 | 建议填 CDN 域名，如 `https://cdn.example.com` |
| `SEAWEEDFS_PRESIGN_EXPIRE_SECONDS` | 开启鉴权时预签名 URL 的有效期（秒） | `3600` | 同左 |
| `RESOURCE_TTL_LONG_SECONDS` | `storage_scene=0` 资源过期时间 | `2592000`（1 个月） | 同左 |
| `RESOURCE_TTL_SHORT_SECONDS` | `storage_scene=1` 资源过期时间 | `7200`（2 小时） | 同左 |
| `DASHSCOPE_API_KEY` | RAG 文本向量化 Key（敏感，`SecretStr`），缺失时上传仍成功但跳过入库 | 空 | 密钥管理服务注入 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` / `EMBEDDING_BATCH_SIZE` | 向量化模型 / 维度 / 单批条数（DashScope 单次上限 25） | `qwen3.7-text-embedding` / `1024` / `10` | 换模型时 `EMBEDDING_DIM` 必须与 Qdrant 建库维度一致 |
| `OCR_ENABLED` / `PDF_SCANNED_MIN_CHARS_PER_PAGE` | 扫描件 OCR 兜底开关 / 判定阈值（平均每页字符数） | `true` / `20` | 纯文本库可关掉兜底，省去外部调用 |
| `OCR_BASE_URL` / `OCR_MODEL` / `OCR_TIMEOUT` | OCR 服务地址 / 模型名 / 单页超时（Key 复用 `DASHSCOPE_API_KEY`） | MaaS `.../compatible-mode/v1` / `vanchin/deepseek-ocr` / `60` | 换网关只改这两项 |
| `OCR_RENDER_DPI` / `OCR_MAX_PAGES` | 渲染清晰度 / 单文档识别页数上限 | `150` / `50` | 页数上限防止大扫描件拖死上传请求 |
| `QDRANT_HOST` / `QDRANT_PORT` | 向量库地址（collection `knowledge_chunks` 不存在时自动创建） | `127.0.0.1` / `6333` | 指向内网向量库 |
| `RAG_INGEST_ENABLED` | 上传文件后是否自动向量化入库 | `true` | `true`（不需要时置 `false`） |
| `RESOURCE_CLEANUP_HOUR` / `RESOURCE_CLEANUP_MINUTE` | 过期清理触发时刻 | `3` / `0`（每天 03:00） | 同左 |
| `REGISTER_RATE_LIMIT` / `REGISTER_RATE_WINDOW` | 注册接口限流配额（次数 / 窗口秒数，按 IP） | `5` / `60` | `5` / `60` |
| `LOGIN_RATE_LIMIT` / `LOGIN_RATE_WINDOW` | 登录接口限流配额（防口令爆破） | `10` / `60` | `10` / `60` |
| `JWT_SECRET_KEY` | JWT 签名密钥（敏感，**任何环境都必填**，长度 >= 32） | 环境变量 / `.env` | 密钥管理服务注入 |
| `JWT_ALGORITHM` | 签名算法（校验时固定该算法，避免 alg 混淆） | `HS256` | `HS256` |
| `JWT_ACCESS_TOKEN_EXPIRE_SECONDS` | 访问令牌有效期 | `1800` | `1800` |
| `JWT_REFRESH_TOKEN_EXPIRE_SECONDS` | 刷新令牌有效期 | `604800` | `604800` |

表中只有 `APP_ENV`、`DOCS_ENABLED` 按环境取不同默认值，其余为代码默认值，`.env.prod.example` 给出了生产建议值。

分层带来的两个硬约束：

- **敏感信息只进配置层**：连接串、密码只允许写在环境变量 / `.env*`，业务代码统一读 `settings`，
  不写死；字段类型是 `SecretStr`，`repr()`、日志、异常里只会出现 `**********`，需要明文时显式调用
  `settings.sqlalchemy_url`。
- **生产不允许带占位符上线**：`APP_ENV=prod` 且 `MYSQL_URL` 未配置时，配置加载阶段直接报错，
  而不是等到第一个请求才失败。

## 日志体系

日志同时写**控制台**和**文件**（`LOG_DIR/LOG_FILE`），文件按大小轮转（默认 10MB × 7 份），
用于留痕与事后排障；目录不存在会自动创建。

```
2026-09-13 01:50:33 | INFO    | app.access:46 | method=GET path=/ status=200 cost=5.2ms client=127.0.0.1
2026-09-13 01:50:36 | WARNING | app.errors:81 | http error path=/docs method=GET status=404
```

级别策略（按“是否需要人介入”分级，避免日志噪音）：

| 级别 | 记录内容 |
| --- | --- |
| `INFO` | 访问日志（每请求一条）、预期内的业务异常（用户不存在、用户名重复等 4xx） |
| `WARNING` | 参数校验失败、路由未命中（可能是调用方契约不一致，值得关注） |
| `ERROR` | 系统异常与未捕获异常，带完整堆栈，用于排障 |

避免多余日志的具体做法：

- **每请求只记一条**访问日志：中间件只记录方法 / 路径 / 状态码 / 耗时 / 客户端地址，不记录请求体与查询串，密码等敏感信息不会落进日志文件
- **关掉 uvicorn 自带的 access log**（否则同一请求会记两遍），启动与异常日志并入统一通道，一起进文件
- **`/docs`、`/redoc`、`/openapi.json`、`/favicon.ico` 不记访问日志**，避免文档请求刷屏
- 异常只打一次堆栈：中间件只留一条访问痕迹，堆栈由全局异常处理器记录
- **幂等配置**：`setup_logging()` 重复调用（如 `uvicorn --reload`）只替换 handler，不会叠加导致同一行日志写多遍
- 第三方库降噪（`passlib`、未开启 `DB_ECHO` 时的 `sqlalchemy.engine`）

## 忽略规则

`.gitignore` 覆盖三类内容，保证敏感信息与运行时产物不入库：

```gitignore
# 环境变量 / 敏感配置：只保留 *.example 模板
.env
.env.*
!.env.example
!.env.dev.example
!.env.prod.example

# 日志：运行时产物
logs/
*.log
*.log.*
```

其余为 Python 常规忽略项（`__pycache__/`、`.venv/`、`build/`、`dist/`）与工具缓存
（`.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`）、编辑器配置（`.idea/`、`.vscode/`）。

## 表结构变更（一键同步）

修改 `app/db/models.py`（新增字段、改类型、加索引等）后，执行：

```bash
uv run python scripts/db_sync.py -m "新增 nickname 字段"
```

脚本依次完成三件事，等价于手动执行 alembic 命令：

```bash
alembic upgrade head                       # 1. 先让数据库追上已有迁移
alembic revision --autogenerate -m "..."   # 2. 按“模型 vs 数据库”差异生成新迁移
alembic upgrade head                       # 3. 应用新迁移
```

- 模型没有变化时不会生成空的迁移文件，只提示“数据库无需变更”
- 生成的迁移脚本位于 `alembic/versions/`，会带上模型差异，请先 review 再提交

分开执行（自定义迁移内容时）：

```bash
uv run alembic revision --autogenerate -m "新增 nickname 字段"
uv run alembic upgrade head      # 应用
uv run alembic downgrade -1      # 回滚一步
```

`alembic/env.py` 从 `app.core.config.settings` 读取连接串，无需在 `alembic.ini` 里重复配置；
autogenerate 已开启 `compare_type`，字段类型变更也能被检测到（`server_default` 变更请手写迁移）。

## 健康检查

`GET /health` 同时校验**服务存活**与**数据库连通性**（执行 `SELECT 1`），供 K8s / LB 探针使用：

| 场景 | 状态码 | 响应体 |
| --- | --- | --- |
| 服务存活且数据库可连 | `200` | `{"status": "ok", "database": "ok"}` |
| 数据库不可达 / 连接串未配置 | `503` | `{"status": "unhealthy", "database": "unreachable"}` |

```bash
curl -i http://127.0.0.1:8000/health
```

稳定性方面的几个约定：

- 数据库异常统一降级为 **503**（不是 500），探针不会因为异常处理链路返回误导性状态码
- 响应只给状态，失败原因（异常类型与消息）**脱敏后写日志**，不对外暴露内部细节
- 引擎带连接超时（`DB_CONNECT_TIMEOUT`，默认 5s），数据库不可达时快速失败，探针不会被拖住
- 探针高频调用，因此 `/health` **不记访问日志**；失败时只留一条 `WARNING`、不打堆栈，避免刷爆日志

实际输出示例：

```
2026-09-13 02:05:14 | WARNING | app.health:56 | health check failed: OperationalError: (2003, "Can't connect to MySQL server on '127.0.0.1' ([Errno 111] Connection refused)")
```

## 注册接口（`POST /auth/register`）

```bash
curl -X POST http://127.0.0.1:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username": "alice", "password": "HZq7mK2p"}'
# 201 {"id": 1, "username": "alice"}
```

路由层只做参数校验与限流，业务逻辑复用 `UserService.create_user`（用户名查重、bcrypt 哈希、
响应过滤密码字段），不重复实现；带版本前缀的 `/api/v1/auth/register` 同样可用。

### 密码策略（`app/core/security.py`）

三条规则需同时满足，任一不通过返回 422 `PARAMETER_ERROR`：

| 规则 | 说明 | 反例 |
| --- | --- | --- |
| 不在弱口令黑名单 | 常见弱口令，含能通过“字母 + 数字”的组合 | `abc123`、`qwerty123` |
| 必须同时含字母和数字 | 避免纯数字 / 纯字母 | `abcdefgh`、`3141592653` |
| 不能包含用户名 | 大小写不敏感的子串匹配 | 用户名 `alice` + 密码 `alice2026x` |

长度沿用请求模型约束（6–72 字符，bcrypt 有效长度上限）；落库前统一 bcrypt 加盐哈希，
明文只存在于请求体内，**不入库、不写日志**。

### 限流（`app/core/rate_limit.py`）

默认 **5 次 / 分钟 / IP**，固定窗口计数，超出返回 429：

```json
{"code": "RATE_LIMITED", "message": "操作过于频繁，请稍后再试", "detail": "ip=203.0.113.7 limit=5/60s"}
```

- 配额由 `REGISTER_RATE_LIMIT` / `REGISTER_RATE_WINDOW` 控制，改配置即可调整
- 只作用于注册接口，其他接口不受影响
- 内存实现、单进程生效：多 worker / 多副本部署需换成 Redis 等共享存储，否则每个进程各算一份配额
- 部署在反向代理后应让 uvicorn 信任代理头（`--proxy-headers --forwarded-allow-ips=...`），
  否则所有请求会共用代理 IP 的配额

## 登录与 JWT 鉴权

### 令牌模型

| 令牌 | 有效期（默认） | 用途 |
| --- | --- | --- |
| access token | 30 分钟 | 访问受保护接口：`Authorization: Bearer <access token>` |
| refresh token | 7 天 | 只用于 `POST /auth/refresh` 换新的访问令牌，不能访问业务接口 |

两类令牌在 payload 里用 `type` 区分，接口只接受对应类型；校验时固定 `algorithms=[HS256]`，
避免 alg 混淆攻击（例如把算法改成 `none`）。`JWT_SECRET_KEY` **只从环境变量 / `.env` 读取**，
未配置或长度不足 32 会直接启动失败，且校验报错不会回显密钥原文（`hide_input_in_errors`）。

### 使用流程

```bash
# 1. 登录，拿到令牌对
curl -X POST http://127.0.0.1:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "alice", "password": "HZq7mK2p"}'
# {"access_token": "eyJ...", "refresh_token": "eyJ...", "token_type": "bearer", "expires_in": 1800}

# 2. 带访问令牌访问受保护接口
curl http://127.0.0.1:8000/auth/me -H 'Authorization: Bearer eyJ...'
# {"id": 1, "username": "alice"}

# 3. 访问令牌过期或需要续期时，用刷新令牌换新的访问令牌
curl -X POST http://127.0.0.1:8000/auth/refresh \
  -H 'Content-Type: application/json' \
  -d '{"refresh_token": "eyJ..."}'
```

鉴权失败按原因返回不同错误码，客户端据此决定“自动刷新”还是“重新登录”：

| 状态码 | code | 场景 | 客户端动作 |
| --- | --- | --- | --- |
| 401 | `INVALID_CREDENTIALS` | 登录时用户名或密码错误 | 提示用户；不区分用户是否存在，避免枚举 |
| 401 | `TOKEN_EXPIRED` | 访问令牌过期 | **自动刷新**后重放原请求 |
| 401 | `TOKEN_INVALID` | 缺少令牌 / 签名错误 / 类型不符 / 用户已删除 | 跳转登录 |

### 自动刷新并重试原请求

`app/clients.py` 提供了调用方实现：捕获 401 `TOKEN_EXPIRED` 后用 refresh token 换新令牌，
再**重放原请求**，调用方无感知；刷新令牌也不可用时才抛出 `RefreshTokenRejectedError` 提示重新登录。
并发请求同时过期时只会触发一次刷新（`asyncio.Lock`）。

```python
async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
    api = RefreshableClient(client)
    await api.login("alice", "HZq7mK2p")
    await api.get("/auth/me")     # access 过期时自动刷新 + 重试，调用方无感知
```

浏览器端等价实现是 axios / fetch 的 401 拦截器：判断 `code == "TOKEN_EXPIRED"` → 调 `/auth/refresh`
→ 更新本地令牌 → 重放原请求（并给刷新加锁，避免并发重复刷新）。

### 给其他接口加鉴权

在路由函数里声明 `CurrentUserDep` 即可（依赖在 `app/api/deps.py`）：

```python
from app.api.deps import CurrentUserDep

@router.get("/orders")
def list_orders(current_user: CurrentUserDep) -> list[Order]:
    ...
```

## 文件上传（`POST /files/upload`，兼容 `/upload/file`）

需登录（`Authorization: Bearer <access token>`）的通用上传接口，multipart 表单字段名为 `file`。
**元数据与文件解耦**：原文件以对象形式存 SeaweedFS，元数据写 MySQL `resources` 表。

| 表单字段 | 取值 | 说明 |
| --- | --- | --- |
| `file` | 文件 | 待上传文件 |
| `storage_scene` | `0`（默认）/ `1` / `2` | 0=长过期（1 个月）、1=短过期（2 小时）、2=只提取内容不存原文件 |
| `upload_purpose` | `0`（默认）/ `1` | 0=普通资源、1=用户头像（仅图片类型会更新 `user.avatar`） |
| `doc_category` | 可空 / `resume` / `study_material` / `general` | 文档分类，**仅文件类型有意义**；不传或传空串按未分类处理，图片 / 音频即使传了也忽略（落 NULL） |

服务端**按内容自动判断类型**，不轻信客户端声明：

| 识别结果 | `resource_type` | 判断依据 | 处理方式 |
| --- | --- | --- | --- |
| 图片 `image` | `1` | 扩展名 / `Content-Type` 初判，再用魔数校验内容 | 传对象；`upload_purpose=1` 时写 `user.avatar` |
| 文档 `document` | `0` | 扩展名 / `Content-Type` 命中文档白名单 | 只传对象，不动用户数据 |
| 音频 `audio` | `2` | 扩展名 / `Content-Type` 命中音频白名单 | 只传对象，不动用户数据 |
| 其他 | - | 不在白名单内 | `415 UNSUPPORTED_FILE_TYPE` |

只要声明是图片，内容魔数就必须匹配（PNG / JPEG / GIF / BMP / WEBP），否则同样按 415 拒绝——
防止把可执行文件改成 `.png` 后写成头像。

### 存储规则

- 对象键：`<user_id>/<内容 MD5>.<扩展名>`，如 `1/9f86d081...c0a.png`（`resources.storage_path` 存的就是它）
- 去重：`UNIQUE(file_hash, user_id)`，即**用户级去重**。代码先按 `(MD5, user_id)` 预查，
  命中直接复用（响应 `deduplicated: true`）；并发写入由唯一索引兜底，冲突后回查先写入的那条
- 文档分类：`doc_category` 只对文件类型生效，取值 `resume` / `study_material` / `general`；
  非法值返回 `422 INVALID_DOC_CATEGORY`。去重命中时，本次**显式带上的分类会覆盖旧值**，
  没带（空）则保留原值
- RAG 入库：**文件类型**落库成功后自动调用 `app/rag/core.py` 的 `ingest_file`
  （解析 -> 清洗 -> 分类 -> 分块 -> DashScope qwen3.7-text-embedding 向量化 -> 写 Qdrant
  -> 分块原文写 MySQL `knowledge_chunks`），
  `doc_category` 原样透传，为空时由 RAG 侧按关键字自动分类；结果通过
  `rag_ingested` / `rag_chunk_count` / `rag_error` 回给前端。
  图片 / 音频、`storage_scene=2`、去重命中的重复上传都不入库
- 分块存储分两份：**Qdrant 只存向量 + 过滤字段**（`user_id` / `doc_category` / `file_name` /
  `chunk_index`，不存原文），**原文存 MySQL `knowledge_chunks`**，两边用入库时生成的
  同一个 UUID 关联（Qdrant point id == `vector_id`）。落库顺序是先 Qdrant 后 MySQL，
  分块写入包在 savepoint 里，失败只回滚分块、不影响 `resources` 落库与上传结果
- PDF 解析：`pypdfium2` 抽文本层；**平均每页字符数 < `PDF_SCANNED_MIN_CHARS_PER_PAGE`（默认 20）**
  判定为扫描件，转 `app/rag/ocr.py` 的 `ocr_pdf` 兜底——按 `OCR_RENDER_DPI`（默认 150）渲染每页为
  PNG，base64 上送视觉 OCR 模型（`OCR_MODEL`，OpenAI 兼容协议），按页序按行拼成全文。
  单个 PDF 最多识别 `OCR_MAX_PAGES`（默认 50）页；`OCR_ENABLED=false` 关闭兜底后扫描件只剩文本层
- 分块：`RecursiveCharacterTextSplitter` 递归切片（`chunk_size=500` / `chunk_overlap=50`），
  分隔符优先级 `\n###` -> `\n##` -> `\n#` -> `\n\n` -> `\n` -> `。！？；，` -> 空格 -> 字符兜底；
  DOCX 的 Heading 1/2/3 会先注入 `#` / `##` / `###`，切块因此能顺着标题与句子边界走
- 向量入库是**覆盖**语义：同一个 `(user_id, file_name)` 重复上传（内容变了）时，
  写库前先按 `user_id` + `file_name` 删除旧数据（Qdrant 删点、MySQL 删分块行），再写本次新数据，
  避免新旧点并存、同一条资料被检索出重复片段。注意顺序是**先删后写**：
  upsert 失败时该文件的旧分块已删除（不会残留重复，重新上传即可恢复）
- `storage_scene=2`：只读内容、不传对象、不写元数据，`extracted_text` 直接返回提取到的文本
- 过期时间：`storage_scene=0` 为 1 个月，`=1` 为 2 小时，写入 `resources.expire_time`
- 安全：只取上传文件名的 basename，防目录穿越；扩展名做字符白名单

```bash
# 登录拿令牌
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "alice", "password": "HZq7mK2p"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

# 上传头像（upload_purpose=1 + 图片类型，才会写 user.avatar）
curl -X POST http://127.0.0.1:8000/files/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./avatar.png" -F "upload_purpose=1"
# 201 {"kind":"image","resource_type":1,"md5":"9f86...c0a","size":20480,"filename":"avatar.png",
#      "path":"1/9f86...c0a.png","avatar_updated":true,"deduplicated":false,
#      "storage_scene":0,"upload_purpose":1,"doc_category":null,"extracted_text":null}

# 上传文档并带分类（仅文件类型有效；不传即未分类）
curl -X POST http://127.0.0.1:8000/files/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./resume.pdf" -F "doc_category=resume"
# 201 {"kind":"document","resource_type":0,...,"doc_category":"resume",
#      "rag_ingested":true,"rag_chunk_count":12,"rag_error":null}

# 只提取内容（不存原文件、不写元数据）
curl -X POST http://127.0.0.1:8000/upload/file \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./note.txt" -F "storage_scene=2"
# 201 {"kind":"document",...,"path":"","extracted_text":"note contents..."}
```

命中该用户已有资源时返回 `"deduplicated": true`，`path` 不变，可直接当作幂等结果使用。

### 错误码

| 状态码 | 错误码 | 场景 |
| --- | --- | --- |
| 401 | `TOKEN_INVALID` / `TOKEN_EXPIRED` | 未带令牌、令牌无效或过期（过期可自动刷新重试） |
| 413 | `FILE_TOO_LARGE` | 超过 `MAX_UPLOAD_BYTES`（默认 10MB），边读边判、不会先落满盘 |
| 415 | `UNSUPPORTED_FILE_TYPE` | 类型不在图片 / 文档 / 音频白名单，或声明是图片但内容对不上 |
| 400 | `EMPTY_FILE` | 上传内容为空 |
| 422 | `UNPROCESSABLE_ENTITY` | `storage_scene` / `upload_purpose` 不在取值范围内 |
| 422 | `INVALID_DOC_CATEGORY` | `doc_category` 不是 `resume` / `study_material` / `general`（空串按未分类处理，不算错） |

> RAG 入库是**增强能力**：没配 `DASHSCOPE_API_KEY`、Qdrant 不可用、文件类型 RAG 不支持（只支持
> `pdf` / `docx` / `txt`）时，上传接口依然 `201`，原因放在 `rag_error`
> （`RAG_DISABLED` / `RAG_UNAVAILABLE` / `RAG_NOT_CONFIGURED` / `UNSUPPORTED_FILE_TYPE` /
> `OCR_FAILED`（扫描件渲染或识别失败）/ `RAG_INGEST_FAILED`）。

> 依赖：文件上传需 `python-multipart==0.0.9`，对象存储用 `boto3`（均已在 `pyproject.toml` 固定，`uv sync` 自动安装）。

## 会话与聊天接口（`/sessions`、`/interviews`）

### 数据语义

- **正文**：`request_text` / `response_text` 是聊天主字段（用户提问 / AI 回答文本）
- **附件**：`request_segments` / `response_segments` 只放附件段，`type` 仅 `file` / `image` / `audio`，
  落库只存 `{"type": "image", "resource_id": 13}`，返回时按 `resource_id` 关联 `resources` 补上 `name` / `url`
- **消息返回**：`status` + `interview_id` 用于前端面试卡片；两者来自该消息开启的面试记录（无面试记录时为 `null`）
- **面试详情**：`GET /interviews/{interview_id}` 按 `interview_id` + 当前 `user_id` 查询，别人的面试返回 404

分页统一返回 `{items, total, page, page_size}`；`page` 从 1 开始，`page_size` 上限 100。
会话列表按 id 倒序（新建在前），消息按时间正序（同一秒内按 id），便于前端顺序渲染。

```jsonc
// GET /sessions/12/messages?page=1&page_size=20
{
  "items": [
    {
      "id": 34, "session_id": 12, "select_model": 4, "request_id": "req-1",
      "request_text": "这道题怎么答",
      "response_text": "这样答",
      "request_segments": [
        {"type": "image", "resource_id": 5, "name": "probe.png", "url": "http://…/test/19/1798….png"}
      ],
      "response_segments": [],
      "status": 1,          // 关联面试的状态：0=进行中，1=已结束，2=异常终止
      "interview_id": 2,    // 该消息开启的面试记录 id
      "created_at": 1789408063
    }
  ],
  "total": 1, "page": 1, "page_size": 20
}
```

### 删除会话的级联动作

`DELETE /sessions/{session_id}` 按以下顺序清理，保证不留孤儿数据 / 孤儿对象：

1. 解析该会话消息的附件段，收集 `resource_id`；再排除「该用户其他会话仍在引用」的资源
2. 删除关联 `interviews`（按 `session_id`）
3. 删除该会话下 `chat_messages`
4. 删除 `sessions` 行
5. 对第 1 步得到的资源：**先删 SeaweedFS 对象，再删 `resources` 元数据**

> 约定：消息由 AI 侧写入（本仓库不提供「发消息」接口），写入时正文放 `*_text`、附件放 `*_segments`；
> 上传接口返回的 `resource_id` 就是附件段里要填的值。

## AI 聊天流式响应（`POST /sessions/{session_id}/stream-chat`，SSE）

用 LangChain 把一次对话拆成三层，每层只做一件事，互不越界（改一层不用动另外两层）：

| 层 | 位置 | 只负责 | 不负责 |
| --- | --- | --- | --- |
| 模型层 | `app/llm/llm.py` | provider -> `BaseChatModel` 实例 | 不拼提示词、不读历史 |
| 提示词层 | `app/prompts/prompt_layer.py` | `ChatPromptTemplate` + LCEL 链路组装 | 不管模板存哪、不管历史怎么取 |
| 记忆层 | `app/memory/memory.py` | 历史轮次构建 + 搜索增强 | 不选模型、不拼最终提示词 |

组装方式（`app/services/stream_chat_service.py`）：

```text
提示词层：公共模板 + 私有模板 + 变量注入结果 + 记忆块 + 行为约束  ->  system
          ChatPromptTemplate(system / history / human) | ChatOpenAI   ->  Runnable
记忆层：  本会话最近 N 轮         -> MessagesPlaceholder("history")
          跨会话检索到的相关片段  -> 拼进 system（【检索到的相关资料】）
模型层：  provider(LLM_PROVIDER) -> ChatOpenAI（DeepSeek 默认，OpenAI 兼容协议）
```

### SSE 事件序列

`meta` -> `delta`* -> `done`（失败则为 `error`），`data` 始终是一行 JSON：

```text
event: meta
data: {"request_id":"9f1c…","session_id":12,"agent_name":"difficulty_learner","scene":"study","prompt_versions":"common=v1,private=v2","history_turns":2,"search_hits":[…],"warnings":[]}

event: delta
data: {"content":"索引"}

event: done
data: {"request_id":"9f1c…","message_id":35,"session_id":12,"answer_length":188,"prompt_versions":"common=v1,private=v2","search_hits":1,"sources":[{"chunk_id":"10cab2fd…","resource_id":13,"chunk_index":0,"file_name":"java.pdf","score":0.7182,"text":"HashMap 底层是数组加链表…"}],"elapsed_ms":1740}
```

`done.sources` 是本轮回答引用的知识片段（问答没引用资料时为空数组），前端据此渲染来源卡片。
下一节「RAG 对话链路」说明这些片段是怎么选出来、怎么注入、怎么落库与回显的。

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/sessions/12/stream-chat \
  -H "Authorization: Bearer $ACCESS_TOKEN" -H 'Content-Type: application/json' \
  -d '{"message":"MySQL 索引优化怎么学","agent_name":"difficulty_learner","use_search":true}'
```

### 请求参数

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `message` | 是 | 本轮提问（1 ~ 8000 字符） |
| `agent_name` | 否 | 智能体名（见 `AGENT_CONFIG`）；留空按会话类型推导（0=学习 -> `difficulty_learner`，1=面试 -> `interview_host`，2=笔记 -> `note_archiver`） |
| `scene` | 否 | 模板场景；留空取该智能体的默认场景 |
| `use_search` | 否 | 是否启用记忆层的搜索增强（默认 `true`） |
| `search_query` | 否 | 检索关键词；留空用 `message` 自动抽取 |

### 两条设计约定

1. **先校验后开流**：SSE 一旦开始下发就改不了 HTTP 状态码，所以「会话不存在 / 越权（404）」
   「智能体不存在（422）」「没配生效模板（404）」都在返回 `StreamingResponse` 之前完成，仍是标准 JSON 错误；
   开流之后的模型故障只能通过 `error` 事件告知前端（此时不落库）
2. **落库用独立会话**：请求级会话在响应结束时才提交，而流式生成器可能在依赖清理之后才结束，
   因此最后一步落库用独立会话工厂写入 `chat_messages`（`request_id` 与 `meta` 事件对齐，便于排查/幂等）；
   客户端中途断开时也会尽力把已生成的部分落库，落库失败只记日志，不影响已经发给用户的回答

### 记忆层的搜索增强

- **历史记忆**：本会话最近 `LLM_HISTORY_TURNS` 轮问答 -> LangChain 消息序列（按轮数与总字符数双重截断）
- **搜索增强**：从该用户**其它会话**的消息正文 / 附件提取文本里做关键词召回 + 打分，取 top-k 片段进 system
  （当前会话的历史已经在对话上下文里，不重复检索）
- 检索后端是 `SearchBackend` 协议，默认 `DatabaseSearchBackend`（纯 SQL `LIKE`，MySQL / SQLite 行为一致）；
  换向量库 / ES 只需实现同一个 `search()` 方法并注入，记忆层其余逻辑不用改
- 检索失败只记 WARNING 并附带一条 `warnings`（SSE `meta` 事件里可见），不让聊天整体不可用

模型层可选 provider：`deepseek`（默认）/ `openai` / `dashscope` / `moonshot` / `zhipu` / `ollama`，
都走 OpenAI 兼容协议，换 provider 只改 `LLM_PROVIDER`（自建网关可配 `LLM_BASE_URL`）；
未配置 Key 时接口返回 500 `LLM_NOT_CONFIGURED`（属服务端配置问题，不静默降级到别的模型）。

## 提示词模板版本管理（`/prompt`）

### 分层思想

```text
公共模板（通用规则，agent_name=NULL，template_type=2）
  + 私有模板（Agent 指令，agent_name 有值，template_type=1）
  + 动态注入的用户信息变量（user_profiles + 运行时信息）
= 完整提示词
```

每个智能体可以独立更新自己的私有模板，公共规则改一版则对所有智能体同时生效；两者分开版本管理，互不干扰。

### 缓存模型（Redis）

- 缓存是一个 Hash：key = `PROMPT_CACHE_KEY`（默认 `prompt:templates:active`）
- field：私有 `{agent}:{scene}`（如 `difficulty_learner:study`），公共 `__common__:{scene}`（如 `__common__:study`）
- value：一行 JSON（`version` / `template_content` / `variables` / `description` / `created_at` / `is_active`）
- 读路径 cache-aside：先读 Redis，未命中查库并回写；Redis 不可用则全程走库（只记 WARNING，功能不受影响）
- 一致性：新建生效版本、一键回滚后**立即刷新对应 field**，保证「生效版本」与缓存一致
- 启动预热：`PROMPT_CACHE_WARMUP=true` 时在后台线程把生效模板灌进 Redis，数据库 / Redis 不可用都不阻塞启动

### 变量注入的三层处理

见 `app/prompts/injector.py`，原始值到最终文本要过三道关：

1. **验证层**：白名单（`ALLOWED_VARIABLES`）+ 类型 + 单值 / 总量长度 + 控制字符 + 注入特征（命中即 422）
2. **转换层**：数组 -> 顿号文本、数字 -> 「3 年」、布尔 -> 是 / 否、空值 -> 「未填写」
3. **填充层**：`re.sub` 单次替换 `{target_job}` -> 实际值；未赋值的占位符原样保留并记入 `warnings`

两个安全要点：**白名单**决定用户注入不了模板作者没登记的东西；**单次替换**保证替换进来的文本不会被二次解析
（用户在字段里写 `{weak_topics}` 这类字面量不会造成「二次占位符注入」）。

### 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/prompt/templates/{agent_name}/{scene}` | 查看该组的公共模板、私有模板与全部历史版本（公共模板用 `__common__` 代指 `agent_name=NULL`） |
| POST | `/prompt/templates` | 新建版本：版本号自动计算，`activate=true` 时立即生效并下线旧版本；`variables` 留空则从正文自动提取 `{变量}` |
| POST | `/prompt/rollback` | 一键回滚：`{"template_id": 3}`，同组其它版本自动置 `is_active=0` 并刷新 Redis |
| GET | `/prompt/config/agents` | `AGENT_CONFIG` 的 7 个智能体映射 + 模型 provider 配置状态 + 缓存配置（`cache_key` / `redis_enabled`） |

```bash
# 新建一版私有模板（立即生效）
curl -X POST http://127.0.0.1:8000/api/v1/prompt/templates \
  -H "Authorization: Bearer $ACCESS_TOKEN" -H 'Content-Type: application/json' \
  -d '{"agent_name":"difficulty_learner","scene":"study","template_content":"目标岗位 {target_job}，薄弱点 {weak_topics}","description":"补充薄弱点关注"}'
# 201 {"id":7,"agent_name":"difficulty_learner","scene":"study","template_type":1,"version":3,"is_active":true,…}

# 一键回滚到 v2
curl -X POST http://127.0.0.1:8000/api/v1/prompt/rollback \
  -H "Authorization: Bearer $ACCESS_TOKEN" -H 'Content-Type: application/json' \
  -d '{"template_id":5}'
# 200 {"agent_name":"difficulty_learner","scene":"study","template_type":1,"active_version":2,"deactivated_versions":[3]}
```

### `AGENT_CONFIG`（7 个智能体）

| 智能体 | 展示名（`label`，即配置里的 `name`） | 默认场景 | `select_model` |
| --- | --- | --- | --- |
| `flow_controller` | 流程总控Agent（默认智能体） | `workflow` | 0 |
| `resume_parser` | 资料&简历解析Agent | `resume` | 3 |
| `quiz_generate_workflow` | 出题题库Agent | `quiz` | 2 |
| `interview_host` | 面试主考官Agent | `interview` | 4 |
| `interview_evaluator` | 面试评测Agent | `evaluation` | 5 |
| `difficulty_learner` | 难点学习Agent | `study` | 1 |
| `note_archiver` | 笔记归档Agent | `note` | 无 |

历史智能体名（`tutor` / `knowledge_explain` / `quiz_coach` / `resume_optimizer` /
`mock_interviewer` / `interview_reviewer` / `study_planner`）在 `LEGACY_AGENT_ALIASES` 里
折叠成上表的现行名字，老调用方仍可照常传 `agent_name`，不会 422。

空库部署后跑 `uv run python scripts/seed_prompt_templates.py --apply` 会按上表写入
「每个场景一条公共模板 + 每个智能体一条私有模板」（可重复执行，已有生效模板的分组自动跳过），
否则 `stream-chat` 会因缺模板返回 404 `PROMPT_TEMPLATE_NOT_FOUND`。

## 接口一览（前缀 `/api/v1`）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/auth/register` | 用户注册（弱密码校验 + 5 次/分钟/IP 限流） |
| POST | `/auth/login` | 登录（签发 access / refresh 令牌） |
| POST | `/auth/refresh` | 用刷新令牌换新的访问令牌 |
| GET | `/auth/me` | 当前登录用户（受保护接口示例，需 Bearer 令牌） |
| POST | `/files/upload`（兼容 `/upload/file`） | 上传文件（需 Bearer 令牌；原文件存 SeaweedFS，元数据存 MySQL） |
| GET | `/sessions` | 会话列表分页（`page` / `page_size`，按当前用户） |
| POST | `/sessions` | 创建会话（`title` + `session_model`） |
| PUT | `/sessions/{session_id}` | 编辑会话标题 |
| DELETE | `/sessions/{session_id}` | 删除会话（级联：消息 / 面试 / 附件资源与对象） |
| GET | `/sessions/{session_id}/messages` | 会话消息分页（正文 + 附件段 + `status`/`interview_id`） |
| POST | `/sessions/{session_id}/stream-chat` | 流式聊天（SSE：`meta` -> `delta`* -> `done`/`error`，需 Bearer 令牌） |
| GET | `/interviews/{interview_id}` | 面试详情（含 `qa_object`，按 `interview_id` + `user_id` 查询） |
| GET | `/prompt/templates/{agent_name}/{scene}` | 查看生效模板（公共 + 私有 + 历史版本，公共用 `__common__`） |
| POST | `/prompt/templates` | 新建模板版本（版本号自增，可选立即生效） |
| POST | `/prompt/rollback` | 一键回滚到指定版本并刷新 Redis 缓存 |
| GET | `/prompt/config/agents` | 智能体映射 / 模型 provider / 提示词缓存配置 |
| POST | `/users` | 新增用户（密码自动哈希） |
| GET | `/users` | 用户列表 |
| GET | `/users/{user_id}` | 用户详情 |
| PATCH | `/users/{user_id}` | 部分更新（用户名 / 密码） |
| DELETE | `/users/{user_id}` | 删除用户 |

## 统一错误处理

所有错误统一返回 `code / message / detail` 结构：

- `code`：机器可读的错误码（如 `USER_NOT_FOUND`），供前端程序判断
- `message`：面向用户的中文提示（如 `用户不存在`）
- `detail`：定位信息（如 `user_id=999`），内部故障时不返回

处理策略（见 `app/core/handlers.py`）：

- 业务异常（`BusinessError`）：4xx，warning 日志，中文可直接展示
- 系统异常（`SystemError` 与未捕获异常）：5xx `SYSTEM_ERROR`，error 日志保留完整堆栈，响应不暴露内部细节
- 参数校验失败：422 `PARAMETER_ERROR`，字段错误汇总在 `detail`
- 未匹配路由等框架异常：按状态码映射中文提示

错误示例：

```json
{"code": "USER_NOT_FOUND", "message": "用户不存在", "detail": "user_id=999"}
```

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/users \
  -H 'Content-Type: application/json' \
  -d '{"username": "alice", "password": "secret123"}'
```

## 运行测试

```bash
# 全量测试（接口 + 数据建模 + 配置分层 + 日志）
uv run pytest

# 冒烟测试：健康检查 + 用户接口主链路（pytest + httpx，最快确认服务可用）
uv run pytest tests/test_smoke.py -v
```

冒烟测试用 `httpx.ASGITransport` 在进程内直连 ASGI 应用，**不需要启动服务、不依赖网络与真实 MySQL**
（数据库依赖被替换为内存 SQLite），适合部署后或改动后第一时间跑一遍：

```text
tests/test_smoke.py::test_health_ok PASSED
tests/test_smoke.py::test_health_returns_503_when_database_unreachable PASSED
tests/test_smoke.py::test_smoke_user_crud_flow PASSED
tests/test_smoke.py::test_smoke_error_response_contract PASSED
tests/test_smoke.py::test_smoke_password_is_hashed PASSED
tests/test_smoke.py::test_smoke_health_is_not_access_logged PASSED
tests/test_smoke.py::test_smoke_uses_real_httpx PASSED
```

`tests/conftest.py` 会在导入应用前用环境变量固定 `APP_ENV=dev` 并把 `LOG_DIR` 指向临时目录，
因此测试用例既不受本机 `.env` 影响，也不会往仓库里写日志。

测试通过 `app.dependency_overrides` 把 `get_db` 换成内存 SQLite，因此**不需要本地启动 MySQL**
也能覆盖接口、服务层与仓储层的完整链路；另有 `tests/test_schema.py` 校验主键、唯一索引、
`create_time` 自动填充，并用 Alembic autogenerate 比对“迁移结果 vs 模型”防止两者不一致；
`tests/test_config.py`、`tests/test_logging.py` 覆盖配置分层、优先级、敏感信息掩码、日志落文件与去重；`tests/test_auth.py` 覆盖注册接口的密码策略、bcrypt 哈希与限流（含 IP 维度隔离、窗口过期恢复）；`tests/test_login.py` 覆盖登录、令牌类型校验、过期令牌、伪造签名、刷新令牌以及“自动刷新并重放原请求”；`tests/test_files.py` 覆盖上传鉴权、图片/文档识别、伪造图片拒绝、MD5 去重、按用户分目录、超限与目录穿越防护。

AI 聊天这条链路另有 6 个测试文件、85 条用例，全部离线（假模型 + 内存 SQLite + 内存 Redis，不连网络、不连真实依赖）：

- `tests/test_llm_layer.py`：模型层的 provider 解析、Key 取值优先级、按 provider 默认模型构造实例、客户端缓存
- `tests/test_prompt_injector.py`：变量注入三层（白名单 / 注入特征 / 长度限制、数组与数字格式化、单次替换防二次注入）
- `tests/test_prompt_template_manager.py`：版本自增与分组独立、同组仅一条生效、Redis 预热 / 命中 / 回写 / 降级、公共+私有拼接、一键回滚
- `tests/test_prompt_api.py`：4 个提示词接口（鉴权、新建版本、查看生效模板、回滚后缓存同步、智能体配置）
- `tests/test_memory.py` / `tests/test_stream_chat.py`：记忆层（历史截断、关键词召回打分、检索失败降级）与 SSE 全链路
  （事件序列、变量与检索内容真的进了 system、历史进消息序列、落库与模型故障走 `error` 事件）
