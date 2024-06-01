import sys
import os
import subprocess

# Add the custom library path
sys.path.append('/mnt/pip_libraries')

# Set environment variables
os.environ['OLLAMA_MODEL_DIR'] = '/mnt/custom_ollama_model_dir'
os.environ['TMPDIR'] = '/mnt/custom_tmp_dir'

# Pull the model
result = subprocess.run(["ollama", "pull", "mistral"], capture_output=True, text=True)
if result.returncode != 0:
    print("Failed to pull the Mistral model.")
    print("Error message:", result.stderr)
    sys.exit(1)

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

# Load the document
loader = PyPDFLoader('redhood.pdf')
pages = loader.load_and_split()

# Initialize the model and embeddings
MODEL = 'mistral'
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# Define the prompt template
template = """
You are a helping chatbot for a tech website and your job is to facilitate people
with their questions. Be polite and Answer the question based on the context provided.
Use the word our not their and don't mention anything about the documents provided for context
Don't reply with based on the context below just make it concise and accurate.
Don't say The document does not provide a specific
If you can't answer the question, just say I can connect you with our team who can assist you further".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# Create the vector store from the documents
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Define the processing chain
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
)

# Interactive Q&A loop with streaming
def interactive_qa():
    print("Enter your questions (type 'exit' to stop):")
    while True:
        question = input("Question: ")
        if question.lower() == 'exit':
            break

        response_stream = chain.stream({"question": question})
        print("Answer: ", end="")
        for chunk in response_stream:
            print(chunk, end="", flush=True)
        print()  # Newline after the response is fully streamed

# Start the interactive Q&A loop
if __name__ == "__main__":
    interactive_qa()
