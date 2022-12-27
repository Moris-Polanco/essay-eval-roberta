import streamlit as st
import transformers

# Load the BERT model
model = transformers.BertModel.from_pretrained('bert-base-uncased')

# Define a tokenizer for the model
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

# Create a function to process the input essay and generate a score
def process_essay(essay):
  # Tokenize the input
  input_ids = tokenizer.encode(essay, return_tensors='pt')
  
  # Generate a score for the essay using the model
  score = model(input_ids)[0]
  
  # Extract the scalar value from the tensor and return it as a float
  return score.item()

# Create the main Streamlit app
def main():
  st.title('Essay Evaluator')
  
  # Get the essay input from the user
  essay = st.text_area('Enter your essay:')
  
  # If the user has entered an essay, process it and display the score
  if essay:
    score = process_essay(essay)
    st.write('Your essay has a score of', score)

if __name__ == '__main__':
  main()
