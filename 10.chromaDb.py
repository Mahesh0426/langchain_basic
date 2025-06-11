from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


# Load a TXT document using TextLoader
loader = TextLoader("data/assignment.txt")
txt_docs = loader.load()


#text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)

#split into chunks
chunks_of_text = text_splitter.split_documents(txt_docs)

#embeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

#store in chroma Db
vector_db = Chroma.from_documents(chunks_of_text,embedding=embedding)


# print("LEN",len(chunks_of_text))
# print("chunks_of_text ",chunks_of_text[2].page_content)  # 2nd chunk of text

print("store in chroma Db",vector_db)

query = "Tell me about Software Piracy and Cybersquatting"
response =  vector_db.similarity_search(query, k=2)

print(response[0].page_content)

# for doc in response:
#     print(doc.page_content)