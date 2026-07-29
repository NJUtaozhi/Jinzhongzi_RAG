# 金种子 — API 开发指南

> 本文档面向前端/后端/全栈开发成员，定义项目的 API 基地址、统一响应格式、错误码规范以及 Docker 部署方式。所有接口均遵循本文档约定；后端实现与前端调用方请以此为准。

---

## 1. API 基地址 (Base URL)

项目在不同阶段使用不同的 Base URL，通过环境变量或前端配置文件切换。

| 阶段 | Base URL | 说明 |
|---|---|---|
| **Docker 部署** | `http://agent:8003` | 容器内通过服务名互访 |
| **本地开发** | `http://localhost:8003` | 本地逐个启动服务 |
| **Mock 阶段** | `http://localhost:3001/api` | 本地 Mock Server（json-server / MSW / express） |

### 1.1 环境变量配置

```env
# Agent 服务
OPENAI_API_KEY=你的API_Key
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4/
VISION_API_URL=http://vision:8001/v1/vision/analyze-face
KNOWLEDGE_API_URL=http://knowledge:8002/v1/knowledge/retrieve

# Frontend
AGENT_API_URL=http://agent:8003/v1/agent/analyze
```

### 1.2 接口完整路径

```
Docker 环境:  http://agent:8003 + /v1/agent/analyze → http://agent:8003/v1/agent/analyze
本地开发:    http://localhost:8003 + /v1/agent/analyze → http://localhost:8003/v1/agent/analyze
```

---

## 2. 统一响应格式

所有接口均返回 `application/json`，**顶层结构固定为三个字段**：

```json
{
  "code": 200,
  "msg": "success",
  "data": { ... }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | `integer` | 业务状态码。`200` 表示成功；其他值参见 [第 3 节 错误码列表](#3-错误码列表) |
| `msg` | `string` | 人类可读的提示信息。成功时固定为 `"success"`；错误时包含面向用户的描述 |
| `data` | `object` / `null` | 响应数据主体。成功时返回业务数据对象；错误时为 `null` |

### 2.1 成功响应示例

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "analysis": {
      "image_emotion": { "dominant_emotion": "sadness", "au12_r_smile_intensity": 0.2, "au04_r_brow_lower": 1.8 },
      "text_sentiment": "negative",
      "text_keywords": ["焦虑", "压力"]
    },
    "decision": "need_knowledge",
    "reply": "听起来你有些焦虑……",
    "advice_source": "从知识库检索到的心理学论文摘要"
  }
}
```

### 2.2 错误响应示例

```json
{
  "code": 40001,
  "msg": "缺少必要参数: image",
  "data": null
}
```

> **约定**：前端应**始终先判断 `code === 200`** 再读取 `data`；若 `code !== 200`，直接取 `msg` 展示错误提示。

---

## 3. 错误码列表

错误码采用 **5 位数字**，按类别分段。

### 3.1 成功码

| 错误码 | 含义 | 说明 |
|---|---|---|
| `200` | 请求成功 | 所有正常返回均使用此码 |

### 3.2 客户端错误 — 参数/请求类 (`400xx`)

| 错误码 | 含义 | 触发场景 |
|---|---|---|
| `40001` | 缺少必要参数 | 请求未传必填字段（如 `image`、`text`、`query` 为空或缺失） |
| `40002` | 参数格式非法 | 参数类型错误或格式不合法（如 `top_k` 传入字符串） |
| `40003` | 参数值超出范围 | 参数值不在允许区间内（如 `top_k` > 10） |

### 3.3 客户端错误 — 业务/数据类 (`400xx`)

| 错误码 | 含义 | 触发场景 |
|---|---|---|
| `40010` | 未检测到人脸 | 上传的图片中没有人脸或人脸不清晰 |
| `40011` | 多人脸检测 | 图片中包含多张人脸（当前仅支持单人分析） |
| `40012` | 图片质量过低 | 分辨率不足、模糊或光照条件差导致无法分析 |
| `40013` | 文本内容为空或无效 | 文本字段仅含空白字符或无法提取有效情绪信号 |

### 3.4 客户端错误 — 认证/鉴权类 (`401xx`)

| 错误码 | 含义 | 触发场景 |
|---|---|---|
| `40100` | 未认证 | 请求缺少认证 Token |
| `40101` | Token 已过期 | 访问 Token 超时失效 |
| `40102` | Token 非法 | Token 签名验证失败或格式错误 |

### 3.5 客户端错误 — 资源/权限类 (`403xx` / `404xx`)

| 错误码 | 含义 | 触发场景 |
|---|---|---|
| `40300` | 无访问权限 | 用户无权访问该资源 |
| `40400` | 资源不存在 | 请求的用户/记录/知识条目不存在 |

### 3.6 服务端错误 (`500xx`)

| 错误码 | 含义 | 触发场景 |
|---|---|---|
| `50000` | 服务器内部错误 | 未知的服务器异常，需查看服务端日志 |
| `50001` | AI 模型调用失败 | 大模型 API 超时、限流或返回异常 |
| `50002` | 知识库检索异常 | 向量数据库连接失败或索引损坏 |
| `50003` | 文件处理失败 | 图片上传后服务端处理异常（如 OSS 写入失败） |
| `50004` | 第三方服务异常 | 依赖的外部 API（如短信、邮件）不可用 |

### 3.7 错误码速查表

| 错误码 | 含义摘要 |
|---|---|
| `200` | 成功 |
| `40001` | 缺少必要参数 |
| `40002` | 参数格式非法 |
| `40003` | 参数值超出范围 |
| `40010` | 未检测到人脸 |
| `40011` | 多人脸检测 |
| `40012` | 图片质量过低 |
| `40013` | 文本内容为空或无效 |
| `40100` | 未认证 |
| `40101` | Token 已过期 |
| `40102` | Token 非法 |
| `40300` | 无访问权限 |
| `40400` | 资源不存在 |
| `50000` | 服务器内部错误 |
| `50001` | AI 模型调用失败 |
| `50002` | 知识库检索异常 |
| `50003` | 文件处理失败 |
| `50004` | 第三方服务异常 |

---

## 4. 接口列表概览

| 方法 | 路径 | 说明 | Content-Type |
|---|---|---|---|
| `POST` | `/v1/agent/analyze` | 多模态情绪分析与决策 | `multipart/form-data` |
| `POST` | `/v1/knowledge/retrieve` | 心理学知识语义检索 | `application/json` |
| `POST` | `/v1/vision/analyze-face` | 面部动作单元（AU）分析 | `multipart/form-data` |

详细请求/响应 Schema 见 `api_spec.yaml`（OpenAPI 3.0 格式）。

---

## 5. Docker 部署

### 5.1 前提条件

- Docker Desktop 已安装并运行

### 5.2 一键启动

```bash
# 构建并启动所有服务
docker-compose up --build

# 后台运行
docker-compose up -d --build

# 停止所有服务
docker-compose down
```

### 5.3 服务健康检查

启动后可通过以下地址验证各服务状态：

| 服务 | 健康检查地址 |
|------|-------------|
| Vision | `http://localhost:8001/health` |
| Knowledge | `http://localhost:8002/health` |
| Agent | `http://localhost:8003/health` |
| Frontend | `http://localhost:8501` |

### 5.4 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看指定服务日志
docker-compose logs -f agent
```

---

## 6. 附录 — 开发约定

1. **所有接口统一使用 `{code, msg, data}` 结构**，不得在前端自行封装一层额外的响应包装。
2. **时间戳**统一使用 ISO 8601 格式 (`2026-07-15T10:30:00+08:00`)。
3. **跨域 (CORS)**：开发阶段后端需允许前端跨域访问；生产阶段由 Nginx 统一处理。
4. **接口文档**以 `api_spec.yaml` 为唯一事实来源（Single Source of Truth），每次接口变更需同步更新该文件。
5. **环境变量**：敏感信息（API Key 等）通过 `.env` 文件配置，不会提交到 Git。
