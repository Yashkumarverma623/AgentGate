import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
import litellm

async def main():
    for model in ["gemini/gemma-4-26b-a4b-it", "gemini/gemma-4-31b-it"]:
        print(f"Testing {model}...")
        try:
            res = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Hi, who are you? Reply in one sentence."}],
                temperature=0.0,
            )
            print(f"SUCCESS {model}: {res.choices[0].message.content}")
        except Exception as e:
            print(f"FAIL {model}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
