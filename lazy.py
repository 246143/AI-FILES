import os
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import hashlib
 
# Streamlit UI setup
st.title("URL Question Answering")
 
# Input URL and Question from User
url = st.text_input("Enter a URL:")
query = st.text_input("Ask a question about the content of the URL:")
 
# Define a simple cache to store FAISS indexes for each URL
cache = {}
 
def get_url_hash(url):
    return hashlib.md5(url.encode()).hexdigest()
 
if url and query:
    try:
        # Set up Hugging Face API Token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your token
 
        # Check if the embeddings for this URL are cached
        url_hash = get_url_hash(url)
        if url_hash in cache:
            db = cache[url_hash]
        else:
            # Load the document from the URL
            loader = UnstructuredURLLoader(urls=[url])
            document = loader.load()
 
            # Check if document is empty
            if not document or len(document) == 0:
                st.write("Error: Could not extract content from the URL. Please check the URL.")
            else:
                # Split the document into smaller chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                docs = text_splitter.split_documents(document)
 
                # Ensure there are chunks to process
                if len(docs) == 0:
                    st.write("Error: No content to process after splitting. Please check the URL.")
                else:
                    # Limit to first 10 chunks to avoid large processing times
                    docs = docs[:10] if len(docs) > 10 else docs
 
                    # Create embeddings and FAISS vector store
                    embeddings = HuggingFaceEmbeddings()
                    db = FAISS.from_documents(docs, embeddings)
 
                    # Cache the FAISS index for future use
                    cache[url_hash] = db
 
                   
 
                    # Load the QA chain with a compatible Hugging Face Hub LLM (text2text-generation)
                    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.8, "max_length": 1024})
                    chain = load_qa_chain(llm, chain_type="stuff")
 
                    # Perform similarity search with the query
                    docsResult = db.similarity_search(query)
 
                    # Check if any relevant documents were found
                    if docsResult:
                        # Generate an answer using the QA chain
                        answer = chain.run(input_documents=docsResult, question=query)
 
                        # Display the answer
                        st.write("### Answer:")
                        st.write(answer)
                    else:
                        # Display a message if no relevant documents were found
                        st.write("### No answer found in URL.")
    except Exception as e:
        st.write(f"An error occurred: {str(e)}")
else:
    # Prompt the user to provide both URL and question
    st.write("### Please provide both the URL and question to proceed.")