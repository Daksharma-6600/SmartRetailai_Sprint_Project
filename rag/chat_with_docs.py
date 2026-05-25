from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from openai import AzureOpenAI
import os

from dotenv import load_dotenv

# =========================
# AZURE OPENAI CONFIG
# =========================


load_dotenv()


client = AzureOpenAI(

    api_key=os.getenv("AZURE_OPENAI_API_KEY"),

    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),

    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# =========================
# LOAD EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# LOAD FAISS DATABASE
# =========================

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Azure RAG Chatbot Ready")

# =========================
# CHAT LOOP
# =========================

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":

        break

    # =========================
    # RETRIEVE DOCUMENTS
    # =========================

    docs = db.similarity_search(
        query,
        k=3
    )

    # =========================
    # CREATE CONTEXT
    # =========================

    context = "\n\n".join(

        [doc.page_content for doc in docs]
    )

    # =========================
    # FINAL PROMPT
    # =========================

    prompt = f"""

    You are a Smart Retail AI Assistant.

    Answer ONLY from the provided context.

    Context:
    {context}

    Question:
    {query}

    """

    # =========================
    # GENERATE RESPONSE
    # =========================

    response = client.chat.completions.create(

        model=DEPLOYMENT_NAME,

        messages=[

            {
                "role": "system",

                "content":
                "You are a helpful AI retail assistant."
            },

            {
                "role": "user",

                "content": prompt
            }
        ],

        temperature=0.3
    )

    answer = response.choices[0].message.content

    print("\nAI Answer:\n")

    print(answer)

    print("\n----------------------")