from locust import HttpUser, task, between, events
import time
import random

# Load prompts
with open("prompts.txt", "r", encoding="utf-8") as f:
    PROMPTS = [line.strip() for line in f if line.strip()]

class VLLMUser(HttpUser):
    host = "http://localhost:8000"  # Or your ngrok URL
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
                        # Register TTFT as a "request" in Locust metrics
                        events.request.fire(
                            request_type="TTFT",
                            name="time_to_first_token",
                            response_time=ttft,
                            response_length=0,
                            context={},  # or {"user": self}
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

# from locust import HttpUser, task, between
# import random
# import time

# # Load prompts from the file
# with open("prompts.txt", "r", encoding="utf-8") as f:
#     PROMPTS = [line.strip() for line in f if line.strip()]

# class VLLMUser(HttpUser):
#     wait_time = between(0.1, 0.5)  # Simulate some think time between requests

#     @task
#     def complete_text(self):
#         # Randomly select a prompt
#         prompt = random.choice(PROMPTS)

#         headers = {"Content-Type": "application/json"}
#         payload = {
#             "model": "microsoft/phi-1_5",  # Ensure this matches your vLLM server
#             "prompt": prompt,
#             "max_tokens": 10,  # Generate no more than 10 tokens
#             "temperature": 0.8,
#             "stream": True
#         }
#         start = time.time()
#         with self.client.post("/v1/completions", json=payload, headers=headers, stream=True, catch_response=True) as resp:
#             for line in resp.iter_lines():
#                 if line:
#                     ttft = time.time() - start
#                     print("⏱️ TTFT:", ttft, "seconds")
#                     break
