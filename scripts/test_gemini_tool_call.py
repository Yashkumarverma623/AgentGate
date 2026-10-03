import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
import litellm

async def main():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "find_user_id_by_email",
                "description": "Find user id by email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "User email address",
                        }
                    },
                    "required": ["email"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are a helpful retail customer service agent. You must look up the user ID before assisting."},
        {"role": "user", "content": "Hi, my email is alice@example.com. Please find my account."},
    ]

    print("Testing tool calling with gemini/gemini-3.1-flash-lite...")
    res = await litellm.acompletion(
        model="gemini/gemini-3.1-flash-lite",
        messages=messages,
        tools=tools,
        temperature=0.0,
    )
    choice = res.choices[0]
    print(f"Message content: {choice.message.content}")
    print(f"Tool calls: {choice.message.tool_calls}")
    print(f"Tokens: {res.usage.prompt_tokens} prompt, {res.usage.completion_tokens} completion")

if __name__ == "__main__":
    asyncio.run(main())
