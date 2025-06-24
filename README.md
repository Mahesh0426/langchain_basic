# PDF Chat Application

This is a Streamlit-based application that allows you to chat with your PDF documents using LangChain and OpenAI's GPT-4. The application uses embeddings to create a semantic search over your PDF content and provides intelligent responses based on the document's context.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install langchain
pip install python-dotenv
pip install -U langchain-openai
pip install -U langchain-community pip install pypdf
pip install langchain-chroma
pip install streamlit

pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

## Running the Application

1. Make sure your virtual environment is activated (if you created one)

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

1. Upload a PDF file using the file uploader on the web interface
2. Wait for the PDF to be processed (the application will create embeddings)
3. Start asking questions about the content of your PDF in the chat interface
4. The AI will respond based on the context from your PDF

## Features

- PDF document processing and chunking
- Semantic search using OpenAI embeddings
- Interactive chat interface
- Chat history preservation during the session
- Streaming responses for better user experience

## Note

- The application uses OpenAI's GPT-4 model for generating responses
- The embeddings are created using OpenAI's text-embedding-3-large model
- The application processes PDFs in chunks of 1000 characters with 200 character overlap
- The retriever fetches the 2 most relevant chunks for each question

## Requirements

The application requires the following Python packages:

- streamlit
- langchain
- langchain-openai
- langchain-community
- langchain-text-splitters
- langchain-core
- faiss-cpu
- python-dotenv
- openai

## this is whole other project

## How to run

1. conda create -n llmap python=3.13 -y
2. conda activate llmap
3. pip install -r requirements.txt
4. pip install python-dotenv

# other way to run

1. python3 -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. pip install python-dotenv
5. pip install langchain
6. pip install -U langchain-openai
7. pip install langchain-groq
8. pip install -U langchain-community
9. pip install pypdf
10. pip install langchain-chroma
11. pip install langserve==0.0.7

## difference betwen venv and conda

| Feature                 | venv                    | conda                            |
| ----------------------- | ----------------------- | -------------------------------- |
| Scope                   | Python only             | Multiple languages               |
| Package Management      | Python packages via pip | Python & non-Python packages     |
| Handles non-Python deps | No                      | Yes                              |
| Installation            | Built into Python       | Requires separate install        |
| Use Case                | Simple Python projects  | Complex, multi-language projects |
| Performance             | Lightweight, fast       | Heavier, more features           |
| Documentation           | Well-documented         | Less well-documented             |
| Community Support       | Active community        | Less active community            |
| Learning Curve          | Easy to learn           | Difficult to learn               |
| Deployment              | Easy to deploy          | Difficult to deploy              |
| Version Control         | Easy to manage          | Difficult to manage              |
| Scalability             | Easy to scale           | Difficult to scale               |

### langchain open ai docs

https://python.langchain.com/v0.1/docs/integrations/platforms/openai/

### langchain-groq docs

https://python.langchain.com/v0.1/docs/integrations/chat/groq/

### langchain prompts

https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html

### chat prompts

https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/

### json output parser

https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/

### data loader

https://python.langchain.com/v0.1/docs/integrations/document_loaders/

## ⚖️ Summary of Use-Cases

| Method              | Input Type      | Output Type          | Purpose                         |
| ------------------- | --------------- | -------------------- | ------------------------------- |
| `embed_query()`     | Single string   | Single vector (list) | Embed a search/query string     |
| `embed_documents()` | List of strings | List of vectors      | Embed multiple documents/chunks |
| `embed_text()`      | Single string   | Single vector (list) | Embed a single document/chunk   |

## ⚖️ Key Differences CharacterTextSplitter and RecursiveCharacterTextSplitter

| Feature                           | `CharacterTextSplitter` | `RecursiveCharacterTextSplitter`                   |
| --------------------------------- | ----------------------- | -------------------------------------------------- |
| **Splitting method**              | Fixed separator         | Recursive fallback through multiple separators     |
| **Preserves semantic structure?** | ❌ Not really           | ✅ Yes                                             |
| **Speed**                         | ✅ Fast                 | ❌ Slightly slower (more logic)                    |
| **Custom splitting hierarchy**    | ❌ No                   | ✅ Yes                                             |
| **Use case**                      | Simple structured text  | Complex, less structured, or natural language text |
