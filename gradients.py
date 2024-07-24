import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 初始化模型和tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# 定义输入提示和目标幻觉回答
input_text = "Who won the 2020 United States presidential election?"
target_text = "Who won the 2020 United States presidential election? Donald Trump"

# 将输入提示和目标幻觉回答转换为模型的输入格式
input_ids = tokenizer.encode(input_text, return_tensors='pt')
target_ids = tokenizer.encode(target_text, return_tensors='pt')

# 确保输入和目标具有相同的长度
max_len = max(input_ids.size(-1), target_ids.size(-1))
input_ids = torch.nn.functional.pad(input_ids, (0, max_len - input_ids.size(-1)), value=tokenizer.eos_token_id)
target_ids = torch.nn.functional.pad(target_ids, (0, max_len - target_ids.size(-1)), value=tokenizer.eos_token_id)

# 获取模型的词嵌入层
embedding_layer = model.transformer.wte

# 将input_ids转换为嵌入向量，并设置requires_grad=True
inputs_embeds = embedding_layer(input_ids)
inputs_embeds.retain_grad()  # 保留梯度信息
print(f"嵌入：{inputs_embeds.size()}")
# 前向传播，计算logits
outputs = model(inputs_embeds=inputs_embeds, labels=target_ids)
logits = outputs.logits

# 获取目标幻觉回答的logits
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = target_ids[..., 1:].contiguous()

# 计算对数似然
loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
log_probs = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
log_probs = log_probs.view(shift_labels.size())

# 计算log p(y | x)
log_p_y_given_x = log_probs.sum()

# 反向传播，计算梯度
log_p_y_given_x.backward()

# 获取第i个token的嵌入向量的梯度
i = 0  # 假设我们关注第一个token
gradients = inputs_embeds.grad[:, i, :]

# 获取第i个token的嵌入向量的梯度
for i in range(inputs_embeds.size(1)):
    gradients = inputs_embeds.grad[:, i, :]
    print(f"Token {i} inputs_embeds shape: {inputs_embeds.shape}")
    print(f"Token {i} gradient shape: {gradients.shape}")
    # print(gradients)
