# `main.py` 流程解释

This `main.py` file implements a FastAPI-based text-to-speech (TTS) API service. It acts as a front-end proxy, receives TTS requests, forwards them to a Gradio backend for audio generation, and caches generated audio.

以下是其主要流程和组件：

1.  **初始化与配置 (`load_dotenv`, `app = FastAPI()`, `MAX_CACHE_SIZE`, `MODEL_PROMPT_MAP`)**
    *   Load environment variables from `.env`, such as the Gradio backend URL (`GRADIO_URL`).
    *   创建一个 FastAPI 应用实例。
    *   配置内存缓存的最大条目数 (`MAX_CACHE_SIZE`)，默认 100。
    *   定义 `MODEL_PROMPT_MAP`，将不同的模型名称映射到对应的参考语音文件路径。

2.  **Gradio client initialization (`@app.on_event("startup")`)**
    *   On FastAPI startup, it will attempt to connect to the Gradio backend.
    *   It will attempt to connect multiple times with a short delay to allow the Gradio client (`gradio_client`) to initialize successfully.
    *   如果连接失败，会发出警告，表示服务可能无法正常工作。

3.  **文本转语音 API 接口 (`@app.post('/v1/audio/speech')`)**
    *   这是核心功能接口，接收 `POST` 请求，请求体是一个 `SpeechRequest` 对象，包含 `model`（模型名称）和 `input`（要转换的文本）。
    *   **缓存逻辑**：
        *   首先，它会根据 `model` 和 `input` 生成一个唯一的哈希值作为缓存键。
        *   尝试从 `in_memory_cache` (一个 `OrderedDict` 实现的 LRU 缓存) 中获取音频数据。
        *   如果缓存命中，则直接返回缓存中的音频数据，并将该条目移动到缓存的末尾（表示最近使用）。
        *   如果缓存未命中，则继续执行语音生成流程。
    *   **参考语音选择**：
        *   根据请求中的 `model` 参数，从 `MODEL_PROMPT_MAP` 中查找对应的参考语音文件路径。
        *   如果找不到，则使用 `DEFAULT_PROMPT_AUDIO_PATH` 作为默认参考语音。
        *   会检查参考语音文件是否存在，如果不存在则抛出 HTTP 400 或 500 错误。
    *   **Calling the Gradio backend**:
    *   Use `gradio_client.predict` to call the Gradio backend `/gen_single` API.
    *   `call_gradio_with_retry` will retry the request several times with a short wait between attempts.
        *   将参考语音文件数据和输入文本传递给 Gradio。
    *   **Handling Gradio response**:
    *   After Gradio returns a result, the service resolves the result path and reads the generated audio file contents.
        *   **缓存更新**：将新生成的音频内容存入 `in_memory_cache`。如果缓存达到 `MAX_CACHE_SIZE` 上限，会移除最旧的（LRU）缓存条目。
        *   成功后，返回 `audio/wav` 格式的音频数据。
        *   会尝试删除 Gradio 生成的临时音频文件。
    *   **错误处理**：捕获各种异常，并抛出相应的 `HTTPException`。

4.  **健康检查接口 (`@app.get('/health')`)**
    *   Provide a simple health check endpoint that returns service status, Gradio client connection state, and in-memory cache info (current and max entries).

5.  **自动启动请求 (`send_startup_request`)**
    *   这是一个在服务启动后自动执行的异步任务。
    *   它会等待几秒钟，然后向 `/v1/audio/speech` 接口发送一个测试请求（使用 `chixiaotu6` 模型和“您好。”文本）。
    *   发送第一次请求后，会等待一小段时间，然后再次发送相同的请求，目的是验证缓存是否正常工作（第二次请求应该命中缓存）。
    *   打印请求结果和状态，用于调试和验证服务启动后的基本功能。

6.  **主程序入口 (`if __name__ == "__main__":`)**
    *   这是 Python 脚本的执行入口。
    *   它使用 `uvicorn.Config` 配置 FastAPI 应用的运行参数（监听地址、端口、日志级别）。
    *   创建一个 `uvicorn.Server` 实例。
    *   使用 `asyncio.create_task` 同时启动 Uvicorn 服务器任务 (`server.serve()`) 和自动发送请求任务 (`send_startup_request()`)。
    *   `asyncio.gather` 等待这两个任务完成（服务器任务会一直运行）。

**总结来说，这个 `main.py` 文件构建了一个高效的文本转语音代理服务，通过以下方式优化用户体验和性能：**
*   **Gradio integration**: Use Gradio for audio generation.
*   **In-memory cache**: An LRU cache avoids regenerating identical responses and reduces backend load.
*   **Retry logic**: Gradio calls are retried to improve robustness.
*   **健康检查**：提供接口监控服务状态。
*   **启动自检**：通过自动发送请求验证服务启动后的基本功能和缓存机制。