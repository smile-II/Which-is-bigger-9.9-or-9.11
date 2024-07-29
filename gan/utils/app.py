from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = Flask(__name__, static_folder='static')
CORS(app)

model_path = "/home/home/txs/sjl/model/glm3"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer once
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.half()

def generate_response(prompt, model, tokenizer, device="cuda"):
    # Tokenize the input text with attention_mask
    model_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Generate the output
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_length=512,  # Ensure the max length is set
            num_return_sequences=1,  # Ensure only one sequence is returned
            no_repeat_ngram_size=2,  # Avoid repeating n-grams
            repetition_penalty=1.2,  # Add repetition penalty to avoid looping
            use_cache=False  # Ensure that cache is not used
        )

    # Decode the output
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['input']
    prompt = f"用户: {user_input}\n助手: "
    response = generate_response(prompt, model, tokenizer, device)
    return jsonify({'response': response})

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
