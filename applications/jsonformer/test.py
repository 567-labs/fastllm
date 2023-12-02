import openai
from openai_function_call import OpenAISchema

openai.api_base = "https://jxnl--jsonformer-fastapi-app-dev.modal.run/v1"


class User(OpenAISchema):
    name: str
    age: str


completion = openai.ChatCompletion.create(
    model="databricks/dolly-v2-3b",
    temperature=0.1,
    stream=False,
    functions=[User.openai_schema],
    function_call={"name": User.openai_schema["name"]},
    messages=[
        {
            "role": "user",
            "content": "Consider the data below: Jason is 10 and John is 30",
        },
    ],
    max_tokens=1000,
)
print(User.from_response(completion))
