# 中文情绪模型离线目录

此目录在 Git 中只保存说明，不提交大模型权重。

在可访问 Hugging Face 的本地电脑执行：

```bash
python scripts/download_sentiment_model.py
```

完成后，本目录应包含 `config.json`、分词器文件以及
`model.safetensors`。将整个目录随项目复制到服务器。Docker Compose
会只读挂载到 `/models/chinese-emotion`，生产环境设置
`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，不会联网下载。

