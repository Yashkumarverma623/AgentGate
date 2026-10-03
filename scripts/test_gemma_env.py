import os
from dotenv import load_dotenv
load_dotenv()
from typing import Optional, List
from google import genai
from tau_bench.envs.user import BaseUserSimulationEnv, UserStrategy
from tau_bench.envs.retail import MockRetailDomainEnv
from tau_bench.types import Action

class GemmaUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str = "gemma-4-31b-it") -> None:
        super().__init__()
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = model.replace("gemini/", "").replace("models/", "")
        self.history: List[str] = []
        self.system_prompt: str = ""
        self.total_cost: float = 0.0

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        return f"You are a customer interacting with an agent.\nInstruction: {instruction}\nRules: Just generate one line at a time to simulate the customer's message. If satisfied, reply '###STOP###'."

    def _generate(self) -> str:
        full_prompt = self.system_prompt + "\n\n" + "\n".join(self.history) + "\nCustomer:"
        res = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )
        text = res.text.strip()
        # Clean up any "Customer:" prefix if model outputs it
        if text.startswith("Customer:"):
            text = text[len("Customer:"):].strip()
        self.history.append(f"Customer: {text}")
        return text

    def reset(self, instruction: Optional[str] = None) -> str:
        self.system_prompt = self.build_system_prompt(instruction=instruction)
        self.history = ["Agent: Hi! How can I help you today?"]
        return self._generate()

    def step(self, content: str) -> str:
        self.history.append(f"Agent: {content}")
        return self._generate()

    def get_total_cost(self) -> float:
        return self.total_cost

def main():
    print("Testing MockRetailDomainEnv with GemmaUserSimulationEnv...")
    env = MockRetailDomainEnv(user_strategy=UserStrategy.HUMAN, task_index=0)
    env.user = GemmaUserSimulationEnv(model="gemma-4-31b-it")

    reset_res = env.reset(task_index=0)
    print(f"Task instruction: {env.task.instruction[:80]}...")
    print(f"Customer opening: '{reset_res.observation}'")

    step_res = env.step(Action(name="respond", kwargs={"content": "I would be glad to help you exchange items. Could you confirm your full name and zip code?"}))
    print(f"Customer reply: '{step_res.observation}'")
    print("✅ Complete environment integration with Gemma 4 user simulator verified!")

if __name__ == "__main__":
    main()
