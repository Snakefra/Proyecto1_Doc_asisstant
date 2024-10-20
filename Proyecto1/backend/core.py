from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

# Crea el objeto de embeddings
embeddings = OpenAIEmbeddings()

# Accede al índice de Pinecone con los embeddings
docsearch = PineconeLangChain.from_existing_index(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# Configura el modelo de lenguaje (OpenAI)
chat = ChatOpenAI(verbose=True, temperature=0)

# Crea la cadena de preguntas y respuestas
qa = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True
)

# Define la función para ejecutar la pregunta
def run_llm(query: str):
    return qa({"query": query})

# Llama la función
if __name__ == "__main__":
    print(run_llm(query="Tell any movie title and the release year"))
    print(run_llm(query="Tell any terror movie title"))
    print(run_llm(query="Tell two disney series title"))
    print(run_llm(query="What actor played the joker 2019 film"))
    print(run_llm(query="Tell one name of a streaming plataform"))
    print(run_llm(query="Tell me the synopsis of the movie The Wild Robot"))

