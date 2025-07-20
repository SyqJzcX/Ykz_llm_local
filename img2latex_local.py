import openai
from qwen_api import LlmApi
from typing import List


def img2latex_local(model: str, image_path: str) -> str:
    """主函数：调用 LLM API 进行图片到 LaTeX 的转换"""
    qwen_local = LlmLocal(
        model=model
    )

    prompt_list = [
        {
            "type": "text",
            "text": """
                你是一个专业的LaTeX公式识别专家。
                请将图片中的数学公式准确转换为LaTeX代码。
                要求：
                1. 只返回标准LaTeX代码，不包含任何解释或说明，可以直接嵌入到HTML文档中
                2. 公式两边不需要加入任何额外的数学环境标记(如 \\[ \\] 或 \\( \\) 等)
                3. 如果公式末尾包含类似 ( 2 - 1 ) 这样的公式编号，请将这部分去掉
                """
        },
        {
            "type": "image_url",
            "image_url": f"{image_path}"
        }
    ]

    answer = qwen_local.get_response(prompt_list)

    return answer


if __name__ == "__main__":
    latex = img2latex_local(
        model="qwen2.5vl:7b",
        image_path='./formula.png'
    )

    print(latex)
