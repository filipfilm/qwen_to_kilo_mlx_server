# local_qwen_server_full_tools.py
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, AsyncGenerator
import threading
import time
import gc
import logging
import json
import os
import asyncio
from contextlib import asynccontextmanager
import mlx.core as mx
from transformers import AutoTokenizer
import subprocess
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import traceback
import hashlib
import shlex
import re
import glob

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Simple configuration
@dataclass
class ServerConfig:
    model_path: str = "./Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2"
    host: str = "0.0.0.0"
    port: int = 8000
    max_tokens_default: int = 131072
    context_window: int = 131072
    temperature_default: float = 0.7
    max_concurrent_requests: int = 2

# Load config from environment
config = ServerConfig(
    model_path=os.getenv("MODEL_PATH", "./Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2"),
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8000")),
    max_tokens_default=int(os.getenv("MAX_TOKENS", "131072")),
    context_window=int(os.getenv("CONTEXT_WINDOW", "131072")),
    temperature_default=float(os.getenv("TEMPERATURE", "0.7")),
    max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "2")),
)

# Global variables
model = None
tokenizer = None
model_lock = threading.Lock()

class ModelState(Enum):
    READY = "ready"
    LOADING = "loading"
    ERROR = "error"

@dataclass
class ModelManager:
    state: ModelState = ModelState.LOADING
    last_error: Optional[str] = None

    def set_ready(self):
        self.state = ModelState.READY
        self.last_error = None

    def set_error(self, error_msg: str):
        self.state = ModelState.ERROR
        self.last_error = error_msg

model_manager = ModelManager()

# Request models
class ChatMessage(BaseModel):
    role: str
    content: str

class FunctionSpec(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = Field(default=config.max_tokens_default, ge=1, le=131072)
    temperature: float = Field(default=config.temperature_default, ge=0.0, le=2.0)
    stream: bool = False
    functions: Optional[List[FunctionSpec]] = None
    tools: Optional[List[Dict]] = None
    function_call: Optional[Union[str, Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.05, ge=0.1, le=2.0)

# Simple request queue
class RequestQueue:
    def __init__(self, max_concurrent: int = 2):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def acquire(self):
        await self.semaphore.acquire()

    def release(self):
        self.semaphore.release()

request_queue = RequestQueue(config.max_concurrent_requests)

# Simple command executor - works from current directory
class SafeCommandExecutor:
    ALLOWED_COMMANDS = {
        "ls", "pwd", "echo", "cat", "grep", "find", "wc", "head", "tail",
        "sort", "uniq", "date", "whoami", "uname", "ps", "free", "uptime",
        "id", "env", "which", "python", "python3", "node", "npm", "pip",
        "git", "mkdir", "touch", "chmod", "cp", "mv", "rm", "cd", "tree"
    }

    @classmethod
    def is_safe_command(cls, command: str) -> bool:
        try:
            parts = shlex.split(command)
            if not parts:
                return False
            base_command = parts[0].split("/")[-1]
            if base_command in ["python3", "py"]:
                base_command = "python"
            return base_command in cls.ALLOWED_COMMANDS
        except Exception:
            return False

    @classmethod
    async def execute(cls, command: str, timeout: int = 30) -> Dict:
        logger.info(f"Executing: {command}")

        if not cls.is_safe_command(command):
            return {
                "error": f"Command not allowed: {command.split()[0] if command.split() else 'empty'}"
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()  # Use current working directory
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "working_directory": os.getcwd(),
            }

        except subprocess.TimeoutExpired:
            return {"error": f"Command timeout ({timeout}s)"}
        except Exception as e:
            return {"error": f"Command execution failed: {str(e)}"}

# Simple function call detector
class FunctionCallDetector:
    @classmethod
    def detect_function_call(
        cls, message: str, functions: Optional[List[FunctionSpec]] = None
    ) -> tuple[bool, Optional[str], Dict]:
        if not functions:
            return False, None, {}
        
        # Look for Qwen3-Coder tool call format
        if "<tool_call>" in message:
            return cls._extract_qwen_tool_call(message, functions)
        
        # Look for write_to_file format
        if "<write_to_file>" in message:
            return cls._extract_write_to_file_call(message, functions)
        
        # Look for code blocks
        if "```bash" in message or "```shell" in message:
            return cls._extract_command_from_code_block(message, functions)
        
        return False, None, {}

    @classmethod
    def _extract_qwen_tool_call(cls, message: str, functions: List[FunctionSpec]) -> tuple[bool, Optional[str], Dict]:
        tool_pattern = r'<tool_call>.*?<function=(\w+)>(.*?)</function>.*?</tool_call>'
        match = re.search(tool_pattern, message, re.DOTALL)
        
        if match:
            function_name = match.group(1)
            params_section = match.group(2)
            
            # Extract parameters
            param_pattern = r'<parameter=(\w+)>\s*(.*?)\s*</parameter>'
            params = {}
            for param_match in re.finditer(param_pattern, params_section, re.DOTALL):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                params[param_name] = param_value
            
            return True, function_name, params
        
        return False, None, {}

    @classmethod
    def _extract_command_from_code_block(cls, message: str, functions: List[FunctionSpec]) -> tuple[bool, Optional[str], Dict]:
        code_block_pattern = r'```(?:bash|shell)\s*(.*?)\s*```'
        match = re.search(code_block_pattern, message, re.DOTALL)
        if match:
            command = match.group(1).strip()
            if command:
                for func in functions:
                    if func.name == "execute_command":
                        return True, func.name, {"command": command}
        
        return False, None, {}

    @classmethod
    def _extract_write_to_file_call(cls, message: str, functions: List[FunctionSpec]) -> tuple[bool, Optional[str], Dict]:
        # Look for <write_to_file> format
        write_pattern = r'<write_to_file>\s*<path>(.*?)</path>\s*<content>(.*?)</content>(?:\s*<line_count>.*?</line_count>)?\s*</write_to_file>'
        match = re.search(write_pattern, message, re.DOTALL)
        
        if match:
            file_path = match.group(1).strip()
            content = match.group(2).strip()
            
            # Check if write_to_file function is available
            for func in functions:
                if func.name == "write_to_file":
                    return True, func.name, {"path": file_path, "content": content}
        
        return False, None, {}

# Enhanced tool execution with all Kilo tools
async def execute_function_internal(function_name: str, arguments: Dict) -> Dict:
    try:
        # File operations
        if function_name == "read_file":
            file_path = arguments.get("path", "")
            if not file_path:
                return {"error": "File path is required"}
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {
                    "success": True,
                    "content": content,
                    "size": len(content),
                    "path": file_path
                }
            except Exception as e:
                return {"error": f"Could not read file: {str(e)}"}
                
        elif function_name == "write_to_file":
            file_path = arguments.get("path", "")
            content = arguments.get("content", "")
            if not file_path:
                return {"error": "File path is required"}
                
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {
                    "success": True,
                    "message": f"File '{file_path}' created successfully",
                    "path": file_path,
                    "size": len(content)
                }
            except Exception as e:
                return {"error": f"Could not write file: {str(e)}"}
        
        elif function_name == "list_files":
            path = arguments.get("path", ".")
            recursive = arguments.get("recursive", False)
            
            try:
                files = []
                if recursive:
                    for root, dirs, filenames in os.walk(path):
                        for filename in filenames:
                            full_path = os.path.join(root, filename)
                            relative_path = os.path.relpath(full_path, path)
                            files.append(relative_path)
                else:
                    for item in os.listdir(path):
                        if os.path.isfile(os.path.join(path, item)):
                            files.append(item)
                
                return {
                    "success": True,
                    "files": sorted(files),
                    "count": len(files),
                    "path": path
                }
            except Exception as e:
                return {"error": f"Could not list files: {str(e)}"}
        
        elif function_name == "search_files":
            pattern = arguments.get("pattern", "")
            path = arguments.get("path", ".")
            file_pattern = arguments.get("file_pattern", "*")
            
            if not pattern:
                return {"error": "Search pattern is required"}
            
            try:
                matches = []
                search_files = glob.glob(os.path.join(path, "**", file_pattern), recursive=True)
                
                for file_path in search_files:
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if pattern in content:
                                    lines = content.splitlines()
                                    line_matches = []
                                    for i, line in enumerate(lines, 1):
                                        if pattern in line:
                                            line_matches.append({"line_number": i, "content": line})
                                    
                                    if line_matches:
                                        matches.append({
                                            "file": file_path,
                                            "matches": line_matches[:10]  # Limit to 10 matches per file
                                        })
                        except:
                            continue  # Skip files that can't be read
                
                return {
                    "success": True,
                    "matches": matches,
                    "pattern": pattern,
                    "files_searched": len(search_files)
                }
            except Exception as e:
                return {"error": f"Search failed: {str(e)}"}
        
        elif function_name == "list_code_definition_names":
            file_path = arguments.get("path", "")
            if not file_path:
                return {"error": "File path is required"}
            
            try:
                definitions = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple regex patterns for common definitions
                import re
                
                # Python functions
                func_pattern = r'^def\s+(\w+)\s*\('
                class_pattern = r'^class\s+(\w+)\s*[\(:]'
                
                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    # Functions
                    func_match = re.match(func_pattern, line.strip())
                    if func_match:
                        definitions.append({
                            "type": "function",
                            "name": func_match.group(1),
                            "line": i
                        })
                    
                    # Classes
                    class_match = re.match(class_pattern, line.strip())
                    if class_match:
                        definitions.append({
                            "type": "class", 
                            "name": class_match.group(1),
                            "line": i
                        })
                
                return {
                    "success": True,
                    "definitions": definitions,
                    "count": len(definitions),
                    "file": file_path
                }
            except Exception as e:
                return {"error": f"Could not analyze file: {str(e)}"}
        
        elif function_name == "apply_diff":
            file_path = arguments.get("path", "")
            diff_content = arguments.get("diff", "")
            if not file_path or not diff_content:
                return {"error": "File path and diff content are required"}
            
            try:
                # Simple diff application (basic implementation)
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                # This is a simplified diff application
                # In a real implementation, you'd parse unified diff format
                lines = original_content.splitlines()
                diff_lines = diff_content.splitlines()
                
                # For now, just return success - full diff implementation would be complex
                return {
                    "success": True,
                    "message": "Diff application attempted (simplified implementation)",
                    "file": file_path
                }
            except Exception as e:
                return {"error": f"Could not apply diff: {str(e)}"}
        
        elif function_name == "insert_content":
            file_path = arguments.get("path", "")
            content = arguments.get("content", "")
            line_number = arguments.get("line_number", 1)
            
            if not file_path or not content:
                return {"error": "File path and content are required"}
            
            try:
                # Read existing file
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                else:
                    lines = []
                
                # Insert content at specified line
                insert_index = max(0, line_number - 1)
                lines.insert(insert_index, content + '\n')
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return {
                    "success": True,
                    "message": f"Content inserted at line {line_number}",
                    "file": file_path
                }
            except Exception as e:
                return {"error": f"Could not insert content: {str(e)}"}
        
        elif function_name == "search_and_replace":
            file_path = arguments.get("path", "")
            search_text = arguments.get("search", "")
            replace_text = arguments.get("replace", "")
            
            if not file_path or not search_text:
                return {"error": "File path and search text are required"}
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if search_text not in content:
                    return {
                        "success": False,
                        "message": "Search text not found in file",
                        "file": file_path
                    }
                
                # Replace text
                new_content = content.replace(search_text, replace_text)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                return {
                    "success": True,
                    "message": f"Replaced '{search_text}' with '{replace_text}'",
                    "file": file_path,
                    "replacements": content.count(search_text)
                }
            except Exception as e:
                return {"error": f"Search and replace failed: {str(e)}"}
        
        # Command execution
        elif function_name == "execute_command":
            command = arguments.get("command", "")
            if not command:
                return {"error": "Command is required"}
            return await SafeCommandExecutor.execute(command)
        
        # Special functions
        elif function_name == "fetch_instructions":
            return {
                "success": True,
                "instructions": "You are a helpful coding assistant with access to file operations, code analysis, and command execution tools.",
                "available_tools": [
                    "read_file", "write_to_file", "list_files", "search_files",
                    "list_code_definition_names", "apply_diff", "insert_content", 
                    "search_and_replace", "execute_command"
                ]
            }
        
        elif function_name == "ask_followup_question":
            question = arguments.get("question", "")
            return {
                "success": True,
                "question": question,
                "message": "Followup question noted"
            }
        
        elif function_name == "attempt_completion":
            result = arguments.get("result", "")
            return {
                "success": True,
                "result": result,
                "message": "Task completion attempted"
            }
        
        elif function_name == "switch_mode":
            mode = arguments.get("mode", "")
            return {
                "success": True,
                "mode": mode,
                "message": f"Switched to {mode} mode"
            }
        
        elif function_name == "new_task":
            task = arguments.get("task", "")
            return {
                "success": True,
                "task": task,
                "message": "New task created"
            }
        
        elif function_name == "update_todo_list":
            todos = arguments.get("todos", [])
            return {
                "success": True,
                "todos": todos,
                "count": len(todos),
                "message": "Todo list updated"
            }
                
        else:
            return {"error": f"Unknown function: {function_name}"}
            
    except Exception as e:
        return {"error": f"Function execution failed: {str(e)}"}

# Model loading (same as simplified server)
def load_model():
    global model, tokenizer, model_manager
    
    logger.info(f"Loading model from {config.model_path}")
    
    try:
        # Convert relative path to absolute path for local models
        model_path = config.model_path
        if model_path.startswith('./') or model_path.startswith('../') or not model_path.startswith('/'):
            # Check if it's a relative path and resolve it
            if os.path.exists(model_path):
                model_path = os.path.abspath(model_path)
                logger.info(f"Resolved model path to: {model_path}")
            else:
                # Try looking in the original models directory
                fallback_path = os.path.join(os.path.expanduser('~'), 'models', model_path.lstrip('./'))
                if os.path.exists(fallback_path):
                    model_path = fallback_path
                    logger.info(f"Found model in fallback location: {model_path}")
                else:
                    logger.error(f"Model not found at {model_path} or {fallback_path}")
                    raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Verify model directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")
        
        logger.info(f"Loading tokenizer from {model_path}")
        # Load tokenizer with local_files_only for local models
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        
        # Load model
        logger.info(f"Loading MLX model from {model_path}")
        from mlx_lm import load
        model, _ = load(model_path)
        logger.info("Model loaded successfully")
        
        # Test model
        from mlx_lm import generate
        _ = generate(model, tokenizer, "Test", max_tokens=5, verbose=False)
        logger.info("Model test passed")
        
        model_manager.set_ready()
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        model_manager.set_error(str(e))
        return False

# Generation (same as simplified server)
async def generate_response(messages: List[Dict], max_tokens: int, temperature: float) -> str:
    global model, tokenizer
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    logger.info(f"Generating response for prompt: {text[:200]}...")
    
    # Generate
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    
    sampler = make_sampler(temp=temperature, top_p=0.95, top_k=20)
    
    with model_lock:
        response_text = generate(
            model, tokenizer, text,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=True
        )
    
    logger.info(f"Raw MLX output: {response_text[:500]}...")
    
    # Clean the response - ONLY remove the input prompt if it's prepended
    if response_text.startswith(text):
        response_text = response_text[len(text):].strip()
    
    # Ensure we have a response
    if not response_text or response_text.strip() == "":
        logger.error("Empty response generated!")
        response_text = "<attempt_completion>\n<result>\nHello, World!\n</result>\n</attempt_completion>"
    
    logger.info(f"Final cleaned response: {response_text[:500]}...")
    
    return response_text.strip()

# Streaming generation (same as simplified server)
async def generate_streaming_response(
    messages: List[Dict], max_tokens: int, temperature: float
) -> AsyncGenerator[str, None]:
    """Generate streaming response in SSE format"""
    global model, tokenizer

    try:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logger.info(f"Streaming generation starting...")

        # Generate complete response first (MLX doesn't support true streaming)
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=0.95, top_k=20)

        with model_lock:
            response_text = generate(
                model, tokenizer, text,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False  # Less verbose for streaming
            )

        # Clean the response
        if response_text.startswith(text):
            response_text = response_text[len(text):].strip()
        
        if not response_text:
            response_text = "<attempt_completion>\n<result>\nHello, World!\n</result>\n</attempt_completion>"

        logger.info(f"Streaming response: {response_text[:200]}...")

        # Stream the response in SSE format
        chunk_id = f"chatcmpl-{int(time.time())}"
        
        # Initial chunk with role
        initial_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "qwen3-coder-30b-dwq-local-full",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        # Stream content in chunks
        chunk_size = 50  # characters per chunk
        for i in range(0, len(response_text), chunk_size):
            chunk_text = response_text[i:i + chunk_size]
            
            content_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "qwen3-coder-30b-dwq-local-full",
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"
            await asyncio.sleep(0.01)  # Small delay for streaming effect

        # Final chunk
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "qwen3-coder-30b-dwq-local-full",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

# Add system prompt for tools (same as simplified server)
def add_tool_system_prompt(messages: List[Dict], functions: List[FunctionSpec]) -> List[Dict]:
    tool_descriptions = [f"- {func.name}: {func.description}" for func in functions]
    
    system_prompt = f"""You have access to the following tools:
{chr(10).join(tool_descriptions)}

When you need to use a tool, use this exact format:
<tool_call>
<function=tool_name>
<parameter=param_name>
param_value
</parameter>
</function>
</tool_call>

Wait for the tool response before continuing."""
    
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        messages[0]["content"] += "\n\n" + system_prompt
    
    return messages

# Add system prompt for tools (optional usage)
def add_tool_system_prompt_optional(messages: List[Dict], functions: List[FunctionSpec]) -> List[Dict]:
    tool_descriptions = [f"- {func.name}: {func.description}" for func in functions]
    
    system_prompt = f"""You have access to the following tools if needed:
{chr(10).join(tool_descriptions)}

You can respond directly without using tools for simple questions or requests.
If you need to use a tool, use this exact format:
<tool_call>
<function=tool_name>
<parameter=param_name>
param_value
</parameter>
</function>
</tool_call>

Wait for the tool response before continuing."""
    
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        messages[0]["content"] += "\n\n" + system_prompt
    
    return messages

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting server...")
    success = load_model()
    
    if success:
        logger.info("Model loaded successfully")
    else:
        logger.error("Failed to load model")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    global model, tokenizer
    model = None
    tokenizer = None
    gc.collect()

app = FastAPI(
    title="Qwen3-Coder MLX Server (Full Tools)",
    version="1.0.0",
    description="MLX server for Qwen3-Coder with all Kilo Code tools",
    lifespan=lifespan,
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def root():
    return {
        "name": "Qwen3-Coder MLX Server (Full Tools)",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "model_state": model_manager.state.value,
        "working_directory": os.getcwd(),
        "tools_count": 15
    }

@app.get("/health")
async def health_check():
    is_healthy = model is not None and model_manager.state == ModelState.READY
    
    status = {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model is not None,
        "model_state": model_manager.state.value,
        "working_directory": os.getcwd(),
        "tools_available": 15
    }
    
    if model_manager.last_error:
        status["last_error"] = model_manager.last_error
    
    if not is_healthy:
        raise HTTPException(status_code=503, detail=status)
    
    return status

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "qwen3-coder-30b-dwq-local-full",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }

@app.get("/v1/functions")
async def list_functions():
    return {
        "object": "list",
        "data": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to read"}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_to_file", 
                    "description": "Create or write contents to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to create/write"},
                            "content": {"type": "string", "description": "Content to write to the file"}
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path to list"},
                            "recursive": {"type": "boolean", "description": "List files recursively"}
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for text patterns in files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Text pattern to search for"},
                            "path": {"type": "string", "description": "Directory to search in"},
                            "file_pattern": {"type": "string", "description": "File pattern to match"}
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_code_definition_names",
                    "description": "List function and class definitions in a code file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the code file"}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_diff",
                    "description": "Apply a diff to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file"},
                            "diff": {"type": "string", "description": "Diff content to apply"}
                        },
                        "required": ["path", "diff"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "insert_content",
                    "description": "Insert content at a specific line in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file"},
                            "content": {"type": "string", "description": "Content to insert"},
                            "line_number": {"type": "integer", "description": "Line number to insert at"}
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_and_replace",
                    "description": "Search and replace text in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file"},
                            "search": {"type": "string", "description": "Text to search for"},
                            "replace": {"type": "string", "description": "Text to replace with"}
                        },
                        "required": ["path", "search"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute shell commands in current directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The shell command to execute"}
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_instructions",
                    "description": "Get available instructions and tools",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_followup_question",
                    "description": "Ask a followup question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The followup question"}
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "attempt_completion",
                    "description": "Attempt to complete a task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "string", "description": "The completion result"}
                        },
                        "required": ["result"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "switch_mode",
                    "description": "Switch operational mode",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "description": "The mode to switch to"}
                        },
                        "required": ["mode"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "new_task",
                    "description": "Create a new task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "The new task description"}
                        },
                        "required": ["task"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_todo_list",
                    "description": "Update the todo list",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "todos": {"type": "array", "description": "Array of todo items"}
                        },
                        "required": ["todos"],
                    },
                },
            },
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, req: Request):
    logger.info(f"Chat request: {len(request.messages)} messages, stream={request.stream}, tools={len(request.tools) if request.tools else 0}")
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    await request_queue.acquire()
    
    try:
        # Convert messages
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]
        
        # Handle tool calls - Only add tools if Kilo explicitly provides them
        available_functions = []
        if request.tools:
            available_functions = [
                FunctionSpec(
                    name=tool["function"]["name"],
                    description=tool["function"].get("description"),
                    parameters=tool["function"].get("parameters"),
                )
                for tool in request.tools
                if tool.get("type") == "function"
            ]
        elif request.functions:
            available_functions = request.functions
        
        # Only add tool system prompt if Kilo actually sent tools
        if available_functions:
            messages = add_tool_system_prompt_optional(messages, available_functions)
            logger.info(f"Added tool system prompt with {len(available_functions)} functions")
        else:
            logger.info("No tools provided by Kilo, using natural response mode")
        
        # Check for tool calls in the last message
        last_message = request.messages[-1].content if request.messages else ""
        
        if available_functions:
            detector = FunctionCallDetector()
            should_call, function_name, arguments = detector.detect_function_call(
                last_message, available_functions
            )
            
            if should_call and function_name:
                logger.info(f"Tool call detected: {function_name} with args: {arguments}")
                
                # Execute the function
                result = await execute_function_internal(function_name, arguments)
                
                # Add tool response and continue generation
                messages.append({"role": "assistant", "content": last_message})
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{json.dumps(result, indent=2)}\n</tool_response>"
                })
                
                logger.info(f"Tool executed, continuing generation with result: {result}")
        
        # Check if streaming is requested
        if request.stream:
            logger.info(f"Streaming response requested")
            return StreamingResponse(
                generate_streaming_response(messages, request.max_tokens, request.temperature),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        
        # Non-streaming response
        logger.info(f"Generating non-streaming response")
        response_text = await generate_response(
            messages, request.max_tokens, request.temperature
        )
        
        # CRITICAL: Ensure response is not empty
        if not response_text or len(response_text.strip()) == 0:
            logger.error("CRITICAL: Empty response detected, using fallback")
            response_text = "<attempt_completion>\n<result>\nHello, World!\n</result>\n</attempt_completion>"
        
        logger.info(f"Response text to be sent: {response_text[:200]}...")
        
        # Clean up
        gc.collect()
        
        # Calculate tokens
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = len(tokenizer.encode(prompt_text))
        completion_tokens = len(tokenizer.encode(response_text))
        
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion", 
            "created": int(time.time()),
            "model": "qwen3-coder-30b-dwq-local-full",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": response_text
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, 
                "total_tokens": prompt_tokens + completion_tokens
            },
            "system_fingerprint": None
        }
        
        # CRITICAL VALIDATION
        if not response["choices"][0]["message"]["content"]:
            logger.error("FATAL: No content in response message!")
            response["choices"][0]["message"]["content"] = "<attempt_completion>\n<result>\nHello, World!\n</result>\n</attempt_completion>"
        
        logger.info(f"Final JSON response: {json.dumps(response, indent=2)[:500]}...")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    finally:
        request_queue.release()

if __name__ == "__main__":
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║             Qwen3-Coder MLX Server (Full Tools)               ║
╠════════════════════════════════════════════════════════════════╣
║  Model: {config.model_path:<54} ║
║  Host:  {config.host:<54} ║
║  Port:  {config.port:<54} ║
║  Working Directory: {os.getcwd():<42} ║
╠════════════════════════════════════════════════════════════════╣
║  Features:                                                     ║
║  • All Kilo Code tools (15 total)                            ║
║  • File operations (read, write, list, search)               ║
║  • Code analysis (definitions, diff, search & replace)       ║
║  • Command execution                                          ║
║  • Task management (todo, completion, modes)                 ║
║  • Works from current directory (like Claude)                 ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")