import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import WikipediaLoader
from pathlib import Path

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_chatModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


# pdf_path = Path(__file__).parent / "data/Lecture2.pdf"   # another way to get the path
# loader = PyPDFLoader(file_path=str(pdf_path))

# Load a PDF document using PyPDFLoader
loader = PyPDFLoader("data/Lecture2.pdf")
docs = loader.load()
print(docs[1].page_content)

#TXT loader
# loader = TextLoader("data/assignment.txt")
# docs = loader.load()

#CSV loader 
# loader = CSVLoader("data/assignment.csv")
# docs = loader.load()

#.HTML loader
# loader = UnstructuredHTMLLoader("data/assignment.html")
# docs = loader.load()

# for HTML loader bs4 and unstructured libraries are required
#pip install bs4
#pip install unstructured


#wikipedia loader - wikipedia library is required - pip install wikipedia
# loader = WikipediaLoader(query="Tesla", load_max_docs=1)
# docs = loader.load()
# print(docs[0].page_content)

 