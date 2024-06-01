import sys
import os
sys.path.append('/mnt/pip_libraries')
import os

# Set environment variables
os.environ['OLLAMA_MODEL_DIR'] = '/mnt/custom_ollama_model_dir'
os.environ['TMPDIR'] = '/mnt/custom_tmp_dir'

# Now, pull the model
import subprocess

subprocess.run(["ollama", "pull", "mistral-7b"], capture_output=True, text=True)

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch

loader = PyPDFLoader('redhood.pdf')
pages = loader.load_and_split()

MODEL = 'mistral'
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)
# print(pages)

from langchain.prompts import PromptTemplate

template = """
Answer the question based on the context below. Don't reply with based on the context below just make it consise and accorate
And it should be to the point. If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

chain = prompt | model

res = chain.invoke({
  "context": "When i was born everyone called me sam",
  "question": "What is my name?"
})
print(res)

from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()
retriever.invoke("services")

from operator import itemgetter

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
)

questions = [
    "What services you are offering?",
    "When was your company founded?",
    "How can i contact you?",
    "Who is the founder of your company?"
]

for question in questions:
    print(f"Question: {question}")
    print(f"Answer: {chain.invoke({'question': question})}")
    print()

# for s in chain.stream({"question": "What is the purpose of the course?"}):
#   print(s, end="", flush=True)

# MODEL = 'mistral'
# model = Ollama(model=MODEL)
# output = model.invoke("Tell me a joke")
# print(output)
