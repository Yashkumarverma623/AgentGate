import os
from dotenv import load_dotenv
load_dotenv()
from typing import Optional, List, Dict, Any
from litellm import completion
from tau_bench.envs.user import LLMUserSimulationEnv

class GemmaUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str = "gemini/gemma-4-31b-it", provider: str = "gemini") -> None:
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.provider = provider
        self.total_cost = 0.0
        # Don't call self.reset() in init without instruction

    def reset(self, instruction: Optional[str] = None) -> str:
        system_prompt = self.build_system_prompt(instruction=instruction)
        first_user_content = f"{system_prompt}\n\nAgent: Hi! How can I help you today?"
        self.messages = [
            {"role": "user", "content": first_user_content},
        ]
        return self.generate_next_message(self.messages)

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        cleaned = []
        for m in messages:
            if m.get("role") == "system":
                cleaned.append({"role": "user", "content": m["content"]})
            else:
                cleaned.append(m)
        res = completion(model=self.model, custom_llm_provider=self.provider, messages=cleaned, temperature=0.0)
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = (getattr(res, "_hidden_params", {}) or {}).get("response_cost") or 0.0
        return message.content or ""

def main():
    print("Testing GemmaUserSimulationEnv with gemini/gemma-4-31b-it...", flush=True)
    try:
        user = GemmaUserSimulationEnv()
        opening = user.reset("You are Alice in 90210. You want to cancel order #W12345.")
        print(f"Opening statement: '{opening}'", flush=True)
        reply = user.step("I have cancelled order #W12345.")
        print(f"Customer reply: '{reply}'", flush=True)
        print("✅ Customer simulation with Gemma works perfectly!", flush=True)
    except Exception as e:
        import traceback
        print(f"Error: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()
