import streamlit as st
from PyPDF2 import PdfReader                 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()                                                                     # we need to give a google api key , by creating a ,env file.
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):                                                      
    text = ""
    for pdf in pdf_docs:                                                         # the function takes the list of pdf documents, read each pdf , and extracts the text from each page.
        pdf_reader = PdfReader(pdf)                 # it reads the pdfs
        for page in pdf_reader.pages:                 
            text += page.extract_text()                            # it add next pages 
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)                       # these should used to split the input text into managable chunks and result should be stored in variable chuks.
    chunks = text_splitter.split_text(text)
    return chunks                                                                                              

def get_vector_store(text_chunks):                                                                             # its shows list of text chunks         
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")                       # generates embeddings for text chunks.           
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")                                                                 

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])       # how the prompt should be formatted.
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)                                          # chain for question - answer
    return chain 

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")                      # turns text to vector embeddings
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)                    #parameter indicates that the function allows potentially unsafe deserialization of objects, which can be a security risk.
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using our own pdf or local file in your system")

    user_question = st.text_input("Ask a Question from the PDF Files📗")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button📘", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
     main()