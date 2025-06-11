# # splitter and embedding model
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma

# # Load a text document using TextLoader
# loader = TextLoader("data/assignment.txt")
# txt_docs = loader.load()

# # print(docs[0].page_content)

# # Split the text into smaller chunks
# text_splitter = CharacterTextSplitter( 
#     separator="\n\n",
#     chunk_size=1000, 
#     chunk_overlap=200, 
#     length_function=len,
#     is_separator_regex=False
#     )
# texts = text_splitter.create_documents([txt_docs[0].page_content])  


# # Split the text into smaller chunks  | other way of splitting if the text is too long
# # recusive_splitter = RecursiveCharacterTextSplitter(
# #     chunk_size= 1000,
# #     chunk_overlap= 200
# # )
# # texts =recusive_splitter.create_documents([txt_docs[0].page_content]) 

#  #Embedding model
# embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# # print("DOCS",len(txt_docs))
# print("SPLIT_DOCS",(len(texts)))
# # print("CHUNK",(texts[4].page_content))
# # print("CHUNK",(texts[0].page_content))
# # print("CHUNK",(texts[2].page_content))
# # print("EMBEDDING",embedding.embed_query("Hello world!"))
# # print("EMBEDDING",embedding.embed_documents(["Hello world!"]))
# # print("len", len(embedding.embed_documents(["hello world!"])))

# # full_text = txt_docs[0].page_content
# embedding_vector = embedding.embed_query(texts)


# print("Length of full text:", len(texts))
# print("Embedding dimension:", len(embedding_vector))
# print("Embedding vector (first 5 values):", embedding_vector[:5])  # just to preview

# splitter and embedding model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 1. Load the text file
loader = TextLoader("data/assignment.txt")
txt_docs = loader.load()

# 2. Split into smaller chunks
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)
texts = text_splitter.create_documents([txt_docs[0].page_content])

# 3. Extract plain strings from Document objects
text_chunks = [doc.page_content for doc in texts]

# 4. Initialize embedding model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# 5. Embed each chunk
embeddings = embedding.embed_documents(text_chunks)

# 6. Output basic info
print("Number of text chunks:", len(text_chunks))
print("Embedding dimension:", len(embeddings[0]))



 # Show each chunk's size and first few embedding values
for i, (chunk, vector) in enumerate(zip(text_chunks, embeddings)):
    print(f"\nChunk {i+1}:")
    print(f"Text length: {len(chunk)} characters")
    print(f"First 5 embedding values: {vector[:5]}")

