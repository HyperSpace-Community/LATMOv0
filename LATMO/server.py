from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import uuid
import time
import logging
import json 
from fastapi.middleware.cors import CORSMiddleware
from main import process_message, top_level_agent, prioritized_agent
from config import API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize LATMO
latmo = prioritized_agent  # Use the existing prioritized_agent instance

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = process_message(request.message)
        return {
            "response": response,
            "model": "LATMO",
            "usage": {
                "completion_tokens": len(response),
                "total_tokens": len(request.message) + len(response)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Change the route from /models to /chat/models
@app.get("/chat/models")
async def get_models():
    return {
        "models": [{
            "id": "LATMO",
            "name": "LATMO Assistant",
            "description": "Langchain-based Assistant for Task Management and Operations",
            "max_tokens": 4000,
            "tokens_per_request": 4000
        }]
    }

# OpenAI API compatible schemas
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: CompletionUsage

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header or api_key_header.replace("Bearer ", "") != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return api_key_header

@app.post("/v1/chat/completions", dependencies=[Depends(get_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Extract the last message content
        last_message = request.messages[-1].content if request.messages else ""
        logger.info(f"Processing chat completion request with {len(request.messages)} messages")
        
        # Build conversation context
        conversation = ""
        for msg in request.messages[:-1]:  # Process all messages except the last one
            conversation += f"{msg.role}: {msg.content}\n"
        
        # Handle the last message separately
        if "arxiv" in last_message.lower():
            # Strip "arxiv" and convert to proper format
            search_query = last_message.replace("arxiv", "", 1).strip()
            conversation += f"system: Searching arXiv for: {search_query}\n"
        else:
            conversation += f"{request.messages[-1].role}: {last_message}\n"
        
        # Process the message
        response = process_message(conversation)
        
        # Clean up response
        response = response.replace("AI: ", "").replace("LATMO: ", "").strip()
        tokens = len(conversation) + len(response)

        logger.info(f"Successfully generated response with {len(response)} characters")

        return ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            usage=CompletionUsage(
                prompt_tokens=len(conversation),
                completion_tokens=len(response),
                total_tokens=tokens
            )
        )
    except Exception as e:
        logger.error(f"Error processing chat completion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/completions", dependencies=[Depends(get_api_key)])
async def chat_completions(request: ChatCompletionRequest):
    """Alternative endpoint for /chat/completions that mirrors /v1/chat/completions"""
    return await create_chat_completion(request)

# Add model list endpoint for OpenAI API compatibility
@app.get("/v1/models", dependencies=[Depends(get_api_key)])
async def list_models():
    return {
        "data": [{
            "id": "LATMO",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "user",
            "permission": [],
            "root": "LATMO",
            "parent": None,
        }]
    }

@app.get("/models")
async def models_redirect():
    """Redirect /models to /v1/models for compatibility"""
    return await list_models()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
