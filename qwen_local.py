from ollama import chat
from ollama import ChatResponse
import os
from typing import List
import base64
import json


class LlmLocal:
    def __init__(self, model: str):
        # 设置模型
        self.model = model

    def get_response(self, prompt: str, image_path: str) -> str:
        """获取 LLM 响应"""
        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        image_path,
                    ]
                }
            ],
            stream=False  # 设置为 False 以获取完整响应
        )

        # print(response.model_dump_json())

        return response['message']['content']
