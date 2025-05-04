import os
from dotenv import load_dotenv
load_dotenv()

# Debug: check OPENAI_API_KEY presence
env_key = os.getenv("OPENAI_API_KEY")
if env_key:
    print(f"DEBUG: OPENAI_API_KEY is set: {env_key[:8]}...", flush=True)
else:
    print("DEBUG: OPENAI_API_KEY is not set", flush=True)
import openai
from openai import OpenAI


def ensure_openai_key():
    """
    Ensure the OpenAI API key is set in the environment.
    Returns:
        str: the API key
    Raises:
        RuntimeError: if OPENAI_API_KEY is not found.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return key


class QueryEngine:
    """
    Wrapper for sending user questions (transcriptions) to OpenAI's Chat API
    using openai>=1.0 interface, with graceful handling of common API errors.

    Usage:
        engine = QueryEngine(model="gpt-3.5-turbo", temperature=0.3)
        answer = engine.answer("What time is it?", context_text="...")
    """

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.3):
        api_key = ensure_openai_key()
        # Debug: confirm API key is loaded
        print(f"DEBUG: Loaded OPENAI_API_KEY={api_key[:8]}…", flush=True)
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def answer(self, question: str, context_text: str = None) -> str:
        """
        Send the transcribed question (and optional context) to the LLM and return the answer.
        Gracefully handles common errors such as insufficient quota or invalid model.
        """
        messages = [
            {"role": "system", "content": (
                "You are a helpful robot assistant. "
                "Use any provided context to answer user queries clearly and concisely."
            )}
        ]
        if context_text:
            messages.append({"role": "system", "content": f"Context: {context_text}"})
        messages.append({"role": "user", "content": question})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            code = getattr(e, 'code', None)
            if code in ('insufficient_quota', 'rate_limit_exceeded'):
                return "⚠️ Sorry, I’ve hit my usage limit. Please check your OpenAI billing or try again later."
            elif code == 'model_not_found':
                return f"❌ The model '{self.model}' is unavailable."
            else:
                return f"❌ OpenAI API error: {e}"
