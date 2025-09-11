from image_token import get_token, get_cost


# # from image_token import get_token
# # num_tokens = get_token(model_name="gpt-5", path=r"C:\Users\sish5001\OneDrive - NIQ\Documents\GitHub\imagetoken\Images\kitten.png")

# # print(num_tokens)


# # num_tokens = get_token(model_name="gemini-1.5-vision", path=r"C:\Users\sish5001\OneDrive - NIQ\Documents\GitHub\imagetoken\Images\kitten.png")


# model_name = "gpt-4o-mini"

# urls = [
#     "https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.jpeg",
#     "https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.jpg",
#     "https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.png",
# ]
num_tokens = get_token(model_name="gemini-2.5-pro",path="https://media.licdn.com/dms/image/v2/D4D12AQEj4ADRPfqFyw/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1701459621193?e=1760572800&v=beta&t=mQohKIKB5s87hjhfeQb4OB3T4I1DdXeucfaxGNMwHrk")

print("num token non langchain: " , num_tokens )


cost = get_cost(model_name="gemini-2.5-pro",path="https://media.licdn.com/dms/image/v2/D4D12AQEj4ADRPfqFyw/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1701459621193?e=1760572800&v=beta&t=mQohKIKB5s87hjhfeQb4OB3T4I1DdXeucfaxGNMwHrk" , system_prompt_tokens=100 , approx_output_tokens=199)
print("cost  " , cost)


# import base64
# from pathlib import Path
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from image_token import simulate_image_token_cost

# llm = ChatOpenAI(model="gemini-2.5-pro" , api_key = "sdfsdf")

# image_data_url = "https://raw.githubusercontent.com/srinathmkce/imagetoken/main/Images/kitten.jpg"

# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(
#         content=[
#             {"type": "image_url", "image_url": {"url": image_data_url}},
#         ],
#     ),
# ]

# result = simulate_image_token_cost(llm, messages)

# print(result)

