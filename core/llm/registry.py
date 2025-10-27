from .openai_llm import OpenAILLM
from .anthropic_llm import AnthropicLLM
from .gemini_llm import GeminiLLM
from .groq_llm import GroqLLM

def build_llm(label: str, keys: dict):
    if label == "OpenAI / gpt-4o-mini": return OpenAILLM(keys.get("OPENAI_API_KEY",""), "gpt-4o-mini")
    if label == "OpenAI / gpt-4o": return OpenAILLM(keys.get("OPENAI_API_KEY",""), "gpt-4o")
    if label == "Anthropic / Claude 3.5": return AnthropicLLM(keys.get("ANTHROPIC_API_KEY",""), "claude-3-5-sonnet-latest")
    if label == "Gemini / 1.5 Pro": return GeminiLLM(keys.get("GOOGLE_API_KEY",""), "gemini-1.5-pro")
    if label == "Groq / groq-1.0": return GroqLLM(keys.get("GROQ_API_KEY",""), "groq-1.0")
    raise ValueError("Unknown LLM label")
