import os

# Coba gunakan python-dotenv jika terinstall, atau fallback ke native parser jika belum terinstall
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Native fallback untuk membaca file .env tanpa perlu library pihak ketiga
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

    @classmethod
    def get_active_gemini_key(cls, override_key: str = "") -> str:
        if override_key and override_key.strip():
            return override_key.strip()
        return cls.GEMINI_API_KEY

    @classmethod
    def get_active_openai_key(cls, override_key: str = "") -> str:
        if override_key and override_key.strip():
            return override_key.strip()
        return cls.OPENAI_API_KEY
