# 任务 7.1 完成与验收记录

## 已完成

- [x] 同步最新 `main`，包含 `bb8f520`、`09dfbd2`、`a473c64` 及 7.3/7.4。
- [x] 调研并记录三个候选模型，主选 110M 中文六情绪模型。
- [x] 下载完整离线权重到 `models/chinese-emotion`（390.6 MB，本地保留、Git 忽略）。
- [x] 在 Knowledge 容器实现 `POST /v1/text/analyze-sentiment`。
- [x] 输出主标签、多标签、置信度、强度、效价、完整分数、模型名和耗时。
- [x] 标签兼容前端既有 14 类映射。
- [x] Agent 文本地址与 Vision 地址解耦，Vision 不加载文本模型。
- [x] Agent 理解节点以真实模型标签为准，LLM 只做意图和综合表达。
- [x] 模型异常时保留纯文本对话降级，不因感知服务错误中断。
- [x] 单元测试、Agent 集成测试、Vision/状态机/记忆/前端回归通过。
- [x] Docker Compose、环境变量、离线下载脚本、API 文档和 CI 已更新。

## 本地验证结果

| 项目 | 结果 |
|---|---|
| 模型文件 | `model.safetensors`，409,112,520 bytes |
| 真实 CPU 推理 | `fear` + `sadness`，111.35 ms |
| 7.1 新增模型/API/集成测试 | 9 passed |
| Week 6 + Agent + 记忆组合测试 | 44 passed |
| 状态机测试 | 33 passed |
| Vision 测试 | 9 passed |
| 前端测试 | 31 passed |
| API 规范解析 | 通过 |
| Python 编译检查 | 通过 |

## 服务器待验收

以下项目依赖服务器权限和 Docker 环境，本地无法代替：

- [ ] 上传 `models/chinese-emotion/` 到服务器同名目录。
- [ ] 构建新镜像前确认并清理无用旧镜像，保留可回滚版本。
- [ ] `docker compose up -d --build knowledge agent` 后四服务全 healthy。
- [ ] 执行 `bash scripts/verify_week7.sh`。
- [ ] `free -h` / `docker stats --no-stream` 常态总内存低于 7 GB。
- [ ] 浏览器完成文字、图片、量表、多轮记忆联合验收。

模型目录通过只读 bind mount 持久化，重建容器不会重新下载或丢失权重。

