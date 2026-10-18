"""OpenAI-compatible transport with explicit truncation handling and usage logs."""

import json
import os
import time
from pathlib import Path


class Client:
    def __init__(self, url, model, log=None, temperature=0, max_tokens=4096):
        from openai import OpenAI

        self.api = OpenAI(
            base_url=url,
            api_key=os.environ.get("OPENAI_API_KEY", "local"),
            max_retries=0,
            timeout=300,
        )
        self.model = model
        self.log = Path(log) if log else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.events = []

    def call(self, prompt, kind="generation", max_tokens=None):
        for attempt in range(2):
            started = time.perf_counter()
            current = (
                prompt
                if attempt == 0
                else prompt + "\nBe concise. Return the complete requested output."
            )
            try:
                response = self.api.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": current}],
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                event = {
                    "kind": kind,
                    "model": self.model,
                    "attempt": attempt,
                    "seconds": time.perf_counter() - started,
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                }
                self.record(event)
                if choice.finish_reason == "length":
                    continue
                if choice.finish_reason != "stop" or not choice.message.content:
                    raise RuntimeError("Response did not finish with complete textual output")
                return choice.message.content
            except Exception as error:
                self.record(
                    {
                        "kind": kind,
                        "model": self.model,
                        "attempt": attempt,
                        "seconds": time.perf_counter() - started,
                        "error": type(error).__name__,
                    }
                )
                raise
        raise RuntimeError("Output remains truncated after one concise retry")

    def record(self, event):
        self.events.append(event)
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open("a") as stream:
                stream.write(json.dumps(event) + "\n")
