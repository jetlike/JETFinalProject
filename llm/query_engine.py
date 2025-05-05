import os
import base64
from dotenv import load_dotenv
load_dotenv()
import openai
from openai import OpenAI


def ensure_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return key

class QueryEngine:
    def __init__(self, model: str = "gpt-4.1-nano-2025-04-14", temperature: float = 0.3):
        api_key = ensure_openai_key()
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def answer(self, question: str, image: str = None) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful robot assistant. "
                    "Use any provided image to answer user queries clearly and concisely."
                )
            }
        ]
        # image handling
        if image and os.path.exists(image):
            with open(image, "rb") as f:
                img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("ascii")

        data_uri = f"data:image/jpeg;base64,{b64}"
        messages.append({"role": "user", "content": f"Here is the image for reference:\n\n![]({data_uri})"})

        messages.append({"role": "user", "content": question})

        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature)
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            code = getattr(e, "code", None)
            if code in ("insufficient_quota", "rate_limit_exceeded"):
                return "Sorry, I’ve hit my usage limit. Please check your OpenAI billing or try again later."
            elif code == "model_not_found":
                return f"The model '{self.model}' is unavailable."
            else:
                return f"OpenAI API error: {e}"
