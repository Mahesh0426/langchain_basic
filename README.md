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
