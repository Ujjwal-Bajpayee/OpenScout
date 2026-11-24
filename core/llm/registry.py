from .openai_llm import OpenAILLM, fetch_available_openai_models
from .anthropic_llm import AnthropicLLM, fetch_available_anthropic_models
from .gemini_llm import GeminiLLM, fetch_available_gemini_models
from .groq_llm import GroqLLM, fetch_available_groq_models

def build_llm(provider_or_label: str, keys: dict, model: str = ""):
    if "/" in provider_or_label:
        parts = [x.strip() for x in provider_or_label.split("/", 1)]
        prov = parts[0]
        selected_model = model.strip() if model else parts[1]
    else:
        prov = provider_or_label.strip()
        selected_model = model.strip()

    prov_lower = prov.lower()
    if "openai" in prov_lower:
        return OpenAILLM(keys.get("OPENAI_API_KEY", ""), model=selected_model or "gpt-4o-mini")
    if "anthropic" in prov_lower:
        return AnthropicLLM(keys.get("ANTHROPIC_API_KEY", ""), model=selected_model or "claude-3-5-sonnet-latest")
    if "gemini" in prov_lower or "google" in prov_lower:
        return GeminiLLM(keys.get("GOOGLE_API_KEY", ""), model=selected_model or "gemini-1.5-pro")
    if "groq" in prov_lower:
        return GroqLLM(keys.get("GROQ_API_KEY", ""), model=selected_model or "llama-3.3-70b-versatile")
    raise ValueError(f"Unknown LLM provider: {provider_or_label}")
