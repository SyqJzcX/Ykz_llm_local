from ollama import chat
from ollama import ChatResponse
import os
from typing import List
import base64


class LlmLocal:
    def __init__(self, model: str):
        # 设置模型
        self.model = model

    def get_response(self, prompt_list: List[dict]) -> str:
        """获取 LLM 响应"""
        prompt_json = []

        for prompt_list_item in prompt_list:
            if not isinstance(prompt_list_item, dict):
                raise ValueError("Prompt list items must be dictionaries.")

            # print(f"Processing prompt item: {prompt_list_item}")
            # print(prompt_list_item)

            tmp = self.build_prompt(prompt_list_item)
            prompt_json.append(tmp)

        # prompt = json.dumps(
        #     prompt_json,
        #     ensure_ascii=False  # 非 ASCII 字符会直接输出为原始字符，而不是原始序列
        # )

        # print("Final prompt JSON:")
        # print(prompt)

        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_json
                }
            ],
            temperature=0.1,      # 降低随机性，提高确定性
        )

        print(response.model_dump_json())

        return response.choices[0].message.content.strip() if response.choices else ""

    def encode_image(self, image_path):
        """将图片转换为 Base64 编码"""
        if not os.path.exists(image_path):
            raise ValueError(f"Image path '{image_path}' does not exist.")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def save_base64_image(self, base64_str, output_path):
        """将 Base64 编码的字符串保存为图片文件"""
        if not base64_str.startswith("data:"):
            raise ValueError("Base64 string must start with 'data:' prefix.")
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(base64_str))

    def build_prompt(self, prompt_dict: dict) -> dict:
        """构建 Prompt"""
        # 若为文本类型的 Prompt，则直接返回
        if prompt_dict.get("type") == "text":
            # print("Processing text prompt:", prompt_dict)
            return prompt_dict

        # 若为图片类型的 Prompt，则将图片转换为 Base64 编码返回
        elif prompt_dict.get("type") == "image_url":
            image_path = prompt_dict.get("image_url")

            # print(f"Processing image prompt with path: {image_path}")

            # 验证路径有效性
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"Image path '{image_path}' does not exist.")

            # 自动检测图片类型
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif file_ext == '.png':
                mime_type = 'image/png'
            else:
                raise ValueError(f"Unsupported image format: {file_ext}")

            base64_image = self.encode_image(image_path)

            # 调试输出 Base64 编码的图片
            # self.save_base64_image(base64_image, "debug_image.png")

            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            }

        else:
            raise ValueError("Unsupported prompt type. Use 'text' or 'image'.")
