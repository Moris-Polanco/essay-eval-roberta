import torch
import transformers

# Cargar el modelo de Roberta
model = transformers.RobertaModel.from_pretrained('roberta-base')

# Escribir una función para utilizar el modelo
def generate_response(input_text):
  input_ids = torch.tensor(model.encode(input_text)).unsqueeze(0)
  output = model.generate(input_ids)
  response = model.decode(output[0], skip_special_tokens=True)
  return response

# Prueba de la función
input_text = "Hola, ¿cómo estás?"
response = generate_response(input_text)
print(response)
