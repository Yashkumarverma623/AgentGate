import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

prompt = """You are a user interacting with an agent.

Instruction: You are Alice in 90210. You want to cancel order #W12345.

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.

Agent: Hi! How can I help you today?
User:"""

print("Testing direct Google GenAI SDK call to gemma-4-31b-it...")
try:
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt,
    )
    print("Direct response from gemma-4-31b-it:")
    print(response.text)
except Exception as e:
    import traceback
    print("Error:", e)
    traceback.print_exc()

print("\nTesting direct Google GenAI SDK call to gemini-3.1-flash-lite...")
try:
    response2 = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )
    print("Direct response from gemini-3.1-flash-lite:")
    print(response2.text)
except Exception as e:
    print("Error:", e)
