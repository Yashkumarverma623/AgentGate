import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
import litellm

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"API key starts with: {api_key[:6]}... length: {len(api_key)}")

    # Test gemini-3.1-flash-lite with litellm
    test_models = [
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-2.5-flash",
        "gemini/gemma-4-26b",
        "gemini/gemma-2-27b-it",
        "gemini/gemma-2-9b-it",
    ]

    for model in test_models:
        print(f"\n--- Testing model: {model} ---")
        try:
            res = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Reply with 'OK' and nothing else."}],
                temperature=0.0,
            )
            print(f"SUCCESS {model}: {res.choices[0].message.content}")
        except Exception as e:
            print(f"FAIL {model}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
