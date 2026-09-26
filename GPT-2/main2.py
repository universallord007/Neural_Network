from transformers import pipeline, set_seed
generator = pipeline('text-generation', model = 'gpt2')
set_seed(42)
model_output = generator("hello I am a language model", max_length = 30 , num_return_sequences = 5)

with open("model_output.txt", "w" , encodding = "utf-8") as f :
    f.write(model_output)

print("Model's output saved successfully")