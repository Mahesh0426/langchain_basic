
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# App config
st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("📄 Chat with your PDF")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# If a PDF is uploaded and embeddings not yet generated
if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save and load PDF
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create vectorstore with embeddings
        embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstore = FAISS.from_documents(chunks, embedding=embedding)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # Prompt and model pipeline
        template = """
        You are a helpful assistant. Only answer the question using the context below.

        Context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4", temperature=0.7)

        def format_docs(docs):
            return "\n".join([d.page_content for d in docs])

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        # Show chat history
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user", avatar="🧑"):
                st.markdown(chat["question"])
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(chat["answer"])

        # Input box at the bottom
        user_input = st.chat_input("Ask something about the PDF...")

        if user_input:
            # Display user message immediately
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)

            # Stream and display response
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in chain.stream(user_input):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            # Save to chat history
            st.session_state.chat_history.append({
                "question": user_input,
                "answer": full_response
            })
