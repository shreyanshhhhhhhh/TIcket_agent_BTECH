from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"Key loaded: {api_key[:10]}..." if api_key else "No key found in .env")

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents="Say hello in 5 words."
    )
    print("SUCCESS:", response.text)
except Exception as e:
    print("FAILED:", e)