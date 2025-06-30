# Simple Chatbot Application with memory

This is simple chatbot application that uses Langchain to create a chatbot with memory. It uses AI to answer questions based on the context provided by the user.These app can be used for various purpose like customer support, knowledge base, chatbot, providing information about products, services, etc.

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
pip install -U langchain-community
pip install langchain-chroma
pip install streamlit

pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```
