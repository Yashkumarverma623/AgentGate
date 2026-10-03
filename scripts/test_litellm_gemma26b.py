import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
import litellm

async def main():
    prompt = "You are a customer. Instruction: You want to cancel order #123. Rules: one line at a time.\nAgent: Hi! How can I help?\nCustomer:"
    print("Testing litellm.acompletion with gemini/gemma-4-26b-a4b-it...")
    try:
        res = await litellm.acompletion(
            model="gemini/gemma-4-26b-a4b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        print("Success! Content:")
        print(res.choices[0].message.content)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
