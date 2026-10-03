import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from tau_bench.envs.retail.tasks_test import TASKS_TEST

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
instruction = TASKS_TEST[0].instruction
prompt = f"You are a customer interacting with an agent.\nInstruction: {instruction}\nRules: Just generate one line at a time to simulate customer message.\nAgent: Hi! How can I help you today?\nCustomer:"

for model in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]:
    print(f"\n--- Model: {model} ---")
    try:
        res = client.models.generate_content(model=model, contents=prompt)
        print(f"Success ({model}): {res.text.strip()}")
    except Exception as e:
        print(f"Error ({model}): {e}")
