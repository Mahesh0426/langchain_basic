import os
import bs4
import streamlit as st
from dotenv import load_dotenv

from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize streaming LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7, streaming=True)

# --- Streamlit UI setup
st.set_page_config(page_title="RAG Chatbot from URL", layout="centered")
st.title("🌐 RAG Chatbot From URL")
url = st.text_input("Paste a blog/article URL:", placeholder="https://example.com/article")

if url:
    with st.spinner("📄 Loading and processing content..."):
        # 1. Load web content
        try:
            loader = WebBaseLoader(
                web_path=(url,),
        
            )
            docs = loader.load()
        except Exception as e:
            st.error(f"Failed to load page: {e}")
            st.stop()

        # 2. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # 3. Embed and create vector DB
        embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorDB = Chroma.from_documents(documents=chunks, embedding=embedding)
        retriever = vectorDB.as_retriever(search_kwargs={"k": 2})

        # 4. Load prompt from hub
        prompt = hub.pull("rlm/rag-prompt")

        # 5. Format retrieved docs
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 6. RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("✅ Content processed. Ask your questions below.")

        # 7. Chat interface
        user_input = st.text_input("Ask a question based on the page:", key="qa")

        if user_input:
            docs = retriever.invoke(user_input)
            context = format_docs(docs)

            if not context.strip():
                st.warning("I don't know based on the provided information.")
            else:
                with st.spinner("🤖 Thinking..."):
                    response_container = st.empty()
                    full_response = ""
                    for chunk in rag_chain.stream(user_input):
                        full_response += chunk
                        response_container.markdown(full_response)


