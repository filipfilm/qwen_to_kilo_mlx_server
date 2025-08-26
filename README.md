# Qwen3-Coder MLX Server

A lightweight, high-performance server for running Qwen3-Coder models with MLX on Apple Silicon. Features full OpenAI API compatibility, streaming support, and tool calling capabilities optimized for Kilo coding assistant.

## ✨ Features

- 🚀 **Full OpenAI API Compatibility** - Drop-in replacement for OpenAI's chat completions API
- 📡 **Streaming Support** - Server-Sent Events (SSE) for real-time responses (required for Kilo)
- 🛠️ **Tool Calling** - Built-in support for function calls and shell commands
- ⚡ **Apple Silicon Optimized** - Leverages MLX for optimal performance on M1/M2/M3/M4 Macs
- 🔧 **Simple Configuration** - Environment variables and sensible defaults
- 🔒 **Request Management** - Built-in concurrency controls and queue management
- 📝 **Comprehensive Logging** - Detailed logging for debugging and monitoring

## 📋 Requirements

- **Apple Silicon Mac** (M1, M2, M3, or M4)
- **Python 3.8+**
- **macOS 13.0+** (for MLX compatibility)
- **32GB+ RAM recommended** for 30B models (16GB minimum)

## 🚀 Quick Start

### 1. Install MLX (Apple Silicon only)

```bash
# Ensure you have Xcode command line tools
xcode-select --install

# Install MLX
pip install mlx mlx-lm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install mlx-lm>=0.18.0 transformers>=4.40.0 fastapi>=0.100.0 uvicorn[standard]>=0.20.0 pydantic>=2.0.0
```

### 3. Download Model

#### Option 1: From HuggingFace Hub
```bash
# Create models directory
mkdir -p ~/models

# Clone the model repository
cd ~/models
git clone https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2
```

#### Option 2: Use existing model
```bash
export MODEL_PATH="/path/to/your/model"
```

### 4. Run Server

```bash
python local_qwen_server.py
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
python local_qwen_server.py
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2` | Path to the model directory |
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `8000` | Server port |
| `MAX_TOKENS` | `131072` | Maximum tokens per response (set high to avoid truncation) |
| `CONTEXT_WINDOW` | `131072` | Model context window size |
| `TEMPERATURE` | `0.7` | Default sampling temperature |
| `MAX_CONCURRENT_REQUESTS` | `2` | Maximum concurrent requests |

## 🎯 Cursor/Kilo Configuration

### Important: Streaming Requirement
Kilo requires streaming responses even when not explicitly requested. This server automatically handles this requirement.

### Configuration Steps:
1. Open Cursor settings
2. Add custom model:
   - Name: `Qwen3-Coder Local`
   - API URL: `http://localhost:8000`
   - Model: `qwen3-coder-30b-dwq-local`
   - Max tokens: `131072` (optional, prevents truncation)
3. Test with a simple "hello world" prompt

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
  "stream": true,
  "tools": []
}
```

#### Response (Streaming)
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello!"}}]}
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

### Model Information
**GET** `/v1/models`

### Server Health
**GET** `/health`

## 🛠️ Tool Support

The server includes 15 built-in tools organized into categories:

  ### File Operations
  - **read_file** - Read contents of any file
  - **write_to_file** - Create or modify files
  - **list_files** - List directory contents (with recursive option)
  - **search_files** - Search for text patterns across multiple files

  ### Code Analysis & Modification
  - **list_code_definition_names** - Extract function and class definitions
  - **apply_diff** - Apply code diffs and patches
  - **insert_content** - Insert content at specific line numbers
  - **search_and_replace** - Find and replace text within files

  ### System Operations
  - **execute_command** - Run safe shell commands
  - **fetch_instructions** - Get available tools and capabilities

  ### Task & Project Management
  - **ask_followup_question** - Interactive development assistance
  - **attempt_completion** - Mark task completion and results
  - **switch_mode** - Change operational contexts
  - **new_task** - Create and manage development tasks
  - **update_todo_list** - Maintain project todo lists

Allowed commands include: `ls`, `pwd`, `echo`, `cat`, `grep`, `find`, `python`, `git`, `mkdir`, `cp`, `mv`, `rm`, and more.

## 📡 Streaming Example

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

## ⚠️ Known Limitations

- **Large Diffs**: Due to token generation limits, very large diffs (100+ lines) may get truncated. Use `write_to_file` tool for major refactors instead of `apply_diff`.
- **Memory Usage**: 30B models require ~20GB RAM minimum, 32GB+ recommended
- **Generation Speed**: Large responses (8k+ tokens) may take 30-60 seconds
- **MLX Compatibility**: Requires macOS 13.0+ and Apple Silicon
- **Streaming Only**: Some clients (like Kilo) require streaming even for non-streaming requests

## 🔧 Development

### Project Structure
```
qwen-server/
├── local_qwen_server.py     # Main server file
├── README.md                # This file
└── requirements.txt         # Dependencies
```

### Testing

```bash
# Test the server
curl http://localhost:8000/health

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Solution: Verify model path exists and contains config.json
   Check: ls -la $MODEL_PATH
   ```

2. **Kilo Shows "No assistant messages"**
   ```
   Solution: Server requires streaming - this is handled automatically
   Check: Server logs should show "Streaming response requested"
   ```

3. **Out of Memory**
   ```
   Solution: Reduce MAX_CONCURRENT_REQUESTS to 1
   Consider: Using smaller quantized models (4-bit recommended)
   ```

4. **Tool Calls Not Working**
   ```
   Check: Server logs for tool detection
   Verify: Tool definitions in request payload
   ```

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
python local_qwen_server.py
```

## 📊 Performance Tips

- **Quantization**: Use 4-bit quantized models for better memory efficiency
- **Batch Size**: Keep `MAX_CONCURRENT_REQUESTS` at 1-2 for stability
- **Temperature**: Use 0.1-0.3 for code generation, 0.7-1.0 for creative tasks
- **Context**: Monitor context usage to avoid hitting limits

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional tool implementations
- Response caching
- Model switching without restart
- Multi-user session management
- Prometheus metrics integration

## 📄 License

This project is open source under the MIT License. Ensure compliance with Qwen model license terms.

## 🙏 Acknowledgments

- **MLX Team** - Apple Silicon ML framework
- **Qwen Team** - Powerful coding models
- **Cursor/Kilo** - Excellent AI coding assistant
- **FastAPI** - Modern web framework

---

**Note**: This server is designed for local development. For production use, consider additional security measures, authentication, and monitoring.
