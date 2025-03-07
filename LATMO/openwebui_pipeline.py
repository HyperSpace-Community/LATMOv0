from typing import List
from webuiapi import WebUIApi
from main import initialize_latmo, process_message

class LatmoPipeline:
    def __init__(self):
        self.agent = initialize_latmo()
        # Initialize OpenWebUI API (adjust host/port as needed)
        self.webui = WebUIApi(host='127.0.0.1', port=7860)
    
    async def generate_response(self, message: str) -> dict:
        """Generate response using LATMO and format for OpenWebUI"""
        try:
            # Get response from LATMO
            response = process_message(message)
            
            # Format response for OpenWebUI
            return {
                "type": "text",
                "content": response,
                "metadata": {
                    "model": "LATMO",
                    "usage": {
                        "prompt_tokens": len(message),
                        "completion_tokens": len(response),
                        "total_tokens": len(message) + len(response)
                    }
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "content": f"Error generating response: {str(e)}",
                "metadata": {
                    "model": "LATMO",
                    "error": str(e)
                }
            }
    
    def get_available_models(self) -> List[str]:
        """Return available models for OpenWebUI interface"""
        return ["LATMO"]
    
    def get_model_info(self, model_name: str) -> dict:
        """Return model information for OpenWebUI interface"""
        if model_name.upper() == "LATMO":
            return {
                "name": "LATMO",
                "description": "Langchain-based Assistant for Task Management and Operations",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                "capabilities": [
                    "Gmail Integration",
                    "Google Calendar Integration",
                    "Wikipedia Search",
                    "Web Search",
                    "Code Generation",
                    "Text-to-Speech"
                    "arXiv Research Tool"
                ]
            }
        return None

def create_pipeline():
    """Create and return a LATMO pipeline instance"""
    return LatmoPipeline()

if __name__ == "__main__":
    # Test the pipeline
    import asyncio
    
    async def test_pipeline():
        pipeline = create_pipeline()
        response = await pipeline.generate_response("Hello, what can you do?")
        print("Response:", response)
    
    asyncio.run(test_pipeline())
