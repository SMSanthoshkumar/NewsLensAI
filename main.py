import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API key
load_dotenv()

# Check API Key
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OpenAI API key not found. Please add it in .env file")
    st.stop()

# UI
st.title("🧠 RockyBot: News Research Tool")
st.sidebar.title("🔗 Enter News Article URLs")

# URL Inputs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button
process_url_clicked = st.sidebar.button("🚀 Process URLs")

# LLM
llm = ChatOpenAI(temperature=0.7, max_tokens=500)

# Process URLs
if process_url_clicked:
    urls = [url for url in urls if url.strip()]

    if not urls:
        st.error("⚠️ Please enter at least one URL")
    else:
        with st.spinner("🔄 Processing URLs..."):
            try:
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()

                if not data:
                    st.error("⚠️ No content loaded. Check the URLs.")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000,
                    chunk_overlap=200
                )
                docs = text_splitter.split_documents(data)

                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local("faiss_index")

                st.success("✅ URLs processed and stored successfully!")

            except Exception as e:
                st.error(f"❌ Error processing URLs: {e}")

# Query input
query = st.text_input("❓ Ask a question based on the articles:")

if query:
    if not os.path.exists("faiss_index"):
        st.error("⚠️ Please process URLs first before asking questions.")
    else:
        with st.spinner("🤖 Generating answer..."):
            try:
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.load_local(
                    "faiss_index",
                    embeddings,
                    allow_dangerous_deserialization=True
                )

                prompt = ChatPromptTemplate.from_template("""
You are a helpful news research assistant.
Answer the question based only on the provided context.
At the end, list the sources used.

<context>
{context}
</context>

Question: {input}
""")

                combine_chain = create_stuff_documents_chain(llm, prompt)
                chain = create_retrieval_chain(vectorstore.as_retriever(), combine_chain)
                result = chain.invoke({"input": query})

                st.header("📌 Answer")
                st.write(result["answer"])

                if result.get("context"):
                    st.subheader("📚 Sources:")
                    seen = set()
                    for doc in result["context"]:
                        source = doc.metadata.get("source", "Unknown")
                        if source not in seen:
                            st.write(f"- {source}")
                            seen.add(source)

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")