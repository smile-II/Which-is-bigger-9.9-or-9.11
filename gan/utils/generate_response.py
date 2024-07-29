from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/home/home/txs/sjl/model/glm3", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/home/txs/sjl/model/glm3", trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "Which of these two numbers is larger? A: {} B: {}. \n###You must answer only with the single letter 'A' or 'B'. \n###You can only output a letter. \n###You output letter is:", history=[])
print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)
