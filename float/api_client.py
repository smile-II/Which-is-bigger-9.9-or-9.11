from openai import OpenAI

class APIClient:
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat_with_model(self, input_text, temperature=1):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个友好的助手。"},
                {"role": "user", "content": input_text}
            ],
            stream=False,
            temperature=temperature,
        )
        return response.choices[0].message.content

# 示例调用
# client1 = APIClient(api_key="sk-599eb172897344d0ba73ac3548c1e50a", base_url="https://api.deepseek.com", model="deepseek-chat")
# client2 = APIClient(api_key="f2df1c53e85d3045c88abe21f6b75c26.Qd6UHkWnYqQUXMiW", base_url="https://open.bigmodel.cn/api/paas/v4/", model="glm-4-flash")
