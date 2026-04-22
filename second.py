from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

# Custom prompt template
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Extracting text from PDF
def get_pdf_text(docs):
    text = ""
    try:
        for pdf in docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading PDFs: {e}")
    return text

# Converting text to chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    return text_splitter.split_text(raw_text)

# Using embeddings model and FAISS for vectorstore
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    return faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)

# Generating conversation chain
def get_conversationchain(vectorstore):
    llm = ChatOpenAI(temperature=0.2)
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer')
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory
    )

# Generating response from user queries
def handle_question(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    question = st.text_input("Ask question from your document:")
    if question:
        if st.session_state.conversation:
            handle_question(question)
        else:
            st.warning("Please process documents first.")
    
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(docs)
                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversationchain(vectorstore)

if __name__ == '__main__':
    main()
