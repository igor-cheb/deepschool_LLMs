from locust import HttpUser, task, between
import random

# Load prompts from the file
with open("prompts.txt", "r", encoding="utf-8") as f:
    PROMPTS = [line.strip() for line in f if line.strip()]

class VLLMUser(HttpUser):
    wait_time = between(0.05, 0.15)  # Simulate some think time between requests

    @task
    def complete_text(self):
        # Randomly select a prompt
        prompt = random.choice(PROMPTS)

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "unsloth/gemma-2b-bnb-4bit",  # Ensure this matches your vLLM server
            "prompt": prompt,
            "max_tokens": 10,  # Generate no more than 10 tokens
            "temperature": 0.8
        }
        self.client.post("/v1/completions", json=payload, headers=headers)