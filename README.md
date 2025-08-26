# Qwen3-Coder MLX Server (Simplified)

A lightweight, high-performance server for running Qwen3-Coder models with MLX on Apple Silicon. Features full OpenAI API compatibility, streaming support, and tool calling capabilities optimized for coding assistants Kilo.

## ✨ Features

- 🚀 **Full OpenAI API Compatibility** - Drop-in replacement for OpenAI's chat completions API
- 📡 **Streaming Support** - Server-Sent Events (SSE) for real-time responses
- 🛠️ **Tool Calling** - Built-in support for function calls and shell commands
- ⚡ **Apple Silicon Optimized** - Leverages MLX for optimal performance on M1/M2/M3/M4 Macs
- 🔧 **Simple Configuration** - Environment variables and sensible defaults
- 🔒 **Request Management** - Built-in concurrency controls and rate limiting
- 📝 **Comprehensive Logging** - Detailed logging for debugging and monitoring

## 🎯 Designed For

- **Cursor IDE** - Specifically tested with Kilo for tool calling
- **Code Generation** - Optimized for programming tasks and code assistance
- **Local Development** - Run powerful AI models entirely on your machine
- **API Development** - Standard OpenAI-compatible endpoints

## 📋 Requirements

- **Apple Silicon Mac** (M1, M2, M3, or M4)
- **Python 3.8+**
- **macOS 12.0+**
- **32GB+ RAM recommended** for 30B models

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install mlx-lm transformers fastapi uvicorn pydantic
```

### 2. Download Model

Download a compatible Qwen3-Coder model (DWQ format recommended):

```bash
# Example: Download from Hugging Face
huggingface-cli download mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2
```

### 3. Run Server

```bash
python local_qwen_server_simplified.py
```

The server will start on `http://localhost:8000`

## ⚙️ Configuration

Configure the server using environment variables:

```bash
# Model Configuration
export MODEL_PATH="./Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2"
export MAX_TOKENS="131072"
export CONTEXT_WINDOW="131072"
export TEMPERATURE="0.7"

# Server Configuration
export HOST="0.0.0.0"
export PORT="8000"
export MAX_CONCURRENT_REQUESTS="2"

# Run with custom config
python local_qwen_server_simplified.py
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2` | Path to the model directory |
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `8000` | Server port |
| `MAX_TOKENS` | `131072` | Maximum tokens per response |
| `CONTEXT_WINDOW` | `131072` | Model context window size |
| `TEMPERATURE` | `0.7` | Default sampling temperature |
| `MAX_CONCURRENT_REQUESTS` | `2` | Maximum concurrent requests |

## 🔌 API Endpoints

### Chat Completions
**POST** `/v1/chat/completions`

OpenAI-compatible chat completions endpoint with streaming support.

#### Request Body
```json
{
  "messages": [
    {"role": "user", "content": "Hello, world!"}
  ],
  "max_tokens": 131072,
  "temperature": 0.7,
  "stream": false,
  "tools": []
}
```

#### Response (Non-streaming)
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen3-coder-30b-dwq-local",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### Model Information
**GET** `/v1/models`

List available models.

### Server Health
**GET** `/health`

Check server health and model status.

### Tool Execution
**POST** `/v1/chat/completions` (with tools)

Execute shell commands and file operations through tool calls.

## 🛠️ Tool Support

The server includes built-in tools for:

- **Shell Command Execution** - Run terminal commands
- **File Read/Write** - Read and modify files
- **Directory Operations** - List and navigate directories

### Example Tool Call
```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "messages": [
        {"role": "user", "content": "List the files in the current directory"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "execute_shell_command",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        }
    ]
})
```

## 📡 Streaming Support

Enable real-time streaming responses:

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Write a Python function"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data != '[DONE]':
                chunk = json.loads(data)
                content = chunk['choices'][0]['delta'].get('content', '')
                if content:
                    print(content, end='')
```

## 🎯 Cursor Integration

For Cursor IDE users, configure your Kilo settings:

1. Set server URL: `http://localhost:8000`
2. Model: `qwen3-coder-30b-dwq-local`
3. Enable streaming for best experience

The server is specifically optimized for Cursor's tool calling requirements and response format expectations.

## 🔧 Development

### Project Structure
```
qwen-server/
├── local_qwen_server_simplified.py  # Main server file
├── README.md                        # This file
└── requirements.txt                 # Dependencies
```

### Key Components

- **Model Loading** - MLX-optimized model initialization
- **Request Processing** - Concurrent request handling with queue management
- **Tool System** - Function call detection and execution
- **Streaming Engine** - Server-Sent Events implementation
- **Response Formatting** - OpenAI API compatibility layer

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify model path is correct
   - Ensure sufficient RAM (16GB+ recommended)
   - Check MLX compatibility

2. **Tool Calls Not Working**
   - Verify tool definitions in request
   - Check server logs for execution errors
   - Ensure proper permissions for shell commands

3. **Streaming Issues**
   - Confirm `stream: true` in request
   - Check client SSE parsing
   - Verify network connectivity

### Debug Mode
Enable verbose logging:
```bash
export LOG_LEVEL=DEBUG
python local_qwen_server_simplified.py
```

## 📊 Performance Tips

- **Memory Management** - Automatic garbage collection after requests
- **Concurrency** - Adjust `MAX_CONCURRENT_REQUESTS` based on available RAM
- **Model Size** - Use quantized models (4-bit DWQ) for better performance
- **Temperature** - Lower values (0.1-0.3) for code generation, higher (0.7-1.0) for creative tasks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Performance improvements
- New tool implementations
- Documentation updates
- Model compatibility enhancements

## 📄 License

This project is open source. Please ensure compliance with the Qwen model license terms.

## 🙏 Acknowledgments

- **MLX Team** - For the excellent Apple Silicon ML framework
- **Qwen Team** - For the powerful coding models
- **FastAPI** - For the robust web framework
- **OpenAI** - For the API standard

---

**Note**: This server is designed for local development and testing. For production use, consider additional security measures, rate limiting, and monitoring.
