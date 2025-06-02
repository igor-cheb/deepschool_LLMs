from locust import HttpUser, task, between
import random

with open("prompts.txt", "r", encoding="utf-8") as f:
    PROMPTS = [line.strip() for line in f if line.strip()]

class VLLMUser(HttpUser):
    wait_time = between(0.05, 0.15)

    @task
    def complete_text(self):
        prompt = random.choice(PROMPTS)

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "unsloth/gemma-2b-bnb-4bit",
            "prompt": prompt,
            "max_tokens": 10,
            "temperature": 0.8
        }
        self.client.post("/v1/completions", json=payload, headers=headers)