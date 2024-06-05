from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os




app = Flask(__name__)

# Load the fine-tuned model
model_path = os.path.join(os.getcwd(), "trained_model")

print("Model Path:", model_path)


model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generate_structured_response(chapter, num_questions=5):
    prompt = f"Generate questions based on {chapter} chapter"
    input_text = f"Prompt: {prompt}\nQuestion:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=num_questions, num_beams=5)
    questions = [tokenizer.decode(output, skip_special_tokens=True).split('Question: ')[-1].strip() for output in outputs]
    
    structured_response = f"Here is the Answer for given Prompt :\n \n \n"
    for i, question in enumerate(questions):
        structured_response += f"{i+1}. {question}\n \n \n"
    return structured_response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    data = request.json
    chapter = data.get('chapter')
    num_questions = data.get('num_questions', 5)
    
    response = generate_structured_response(chapter, num_questions)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

