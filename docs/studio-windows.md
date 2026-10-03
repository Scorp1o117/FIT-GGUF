# FIT Studio · Windows 本地应用

1. 将整个压缩包解压到可写目录，保留 `FIT-Studio.exe` 和 `_internal` 的相对位置。
2. 双击 `FIT-Studio.exe`。需要 Windows Edge WebView2 Runtime；普通 Windows 11 通常已经安装。
3. 在“模型适配”中读取 llmfit 的硬件适配建议；这不会下载模型。
4. 在“量化工作台”选择源 GGUF、重要性矩阵和 llama.cpp 运行时目录，再依次分析、规划、量化。

运行时目录应包含 `llama-quantize.exe` 和配套 DLL，且支持 FIT 所需的 dry-run 和张量类型覆盖。可从 [llama.cpp 官方发布页](https://github.com/ggml-org/llama.cpp/releases) 获取对应 Windows 版本。应用包不包含模型、重要性矩阵或 llama.cpp。

默认限制源模型不超过 5B 参数；实验请使用小模型。GPU 文件预算按当前空闲单卡显存扣除预留和运行开销估算，不会把 32GB 内存加到 16GB 显存上。

工作区默认保存在 `%LOCALAPPDATA%\FIT-Studio\workspace`。每个任务有独立的日志、记录和产物，重新启动后可以继续查看。关闭应用会取消仍在运行的任务。

文件大小校验通过不代表量化质量评测通过。校准、保真度搜索和档位搜索仍通过 FIT CLI 操作。模型适配的速度是 llmfit 估算，不能当作本机性能实测。

应用只监听 `127.0.0.1`，模型和分析文件在本地处理。点击源模型链接才会打开 Hugging Face 页面。

许可证见 `LICENSE`、`THIRD_PARTY_NOTICES.md` 与 `licenses/`。此包为本地开发验证构建，未签名，也未发布为正式发行版。
