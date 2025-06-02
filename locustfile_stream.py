from locust import HttpUser, task, between, events
import time
import random

with open("prompts.txt", "r", encoding="utf-8") as f:
    PROMPTS = [line.strip() for line in f if line.strip()]

class VLLMUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(0.1, 0.5)

    @task
    def complete_text(self):
        prompt = random.choice(PROMPTS)
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "microsoft/phi-1_5",
            "prompt": prompt,
            "max_tokens": 10,
            "stream": True
        }

        start_time = time.time()
        try:
            with self.client.post("/v1/completions", json=payload, headers=headers, stream=True, catch_response=True) as response:
                for line in response.iter_lines():
                    if line:
                        ttft = (time.time() - start_time) * 1000  # ms
                        events.request.fire(
                            request_type="TTFT",
                            name="time_to_first_token",
                            response_time=ttft,
                            response_length=0,
                            context={}, 
                            exception=None
                        )
                        break
        except Exception as e:
            events.request.fire(
                request_type="TTFT",
                name="time_to_first_token",
                response_time=0,
                response_length=0,
                exception=e,
                context={}
            )
