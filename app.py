import streamlit as st
import pickle
from PyPDF2 import PdfReader
from docx import Document
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import openai
import time

openai.api_key = st.secrets["OPENAI_API_KEY"]

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

def main():
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.header("Chat to a Document 💬👨🏻‍💻🤖")
    
    # upload a PDF or Word file
    file = st.file_uploader("Upload your PDF or Word document (just one for now)", type=['pdf', 'docx'])

    # Add a text area for user to paste text
    user_text = st.text_area("Or paste your text here:")

        # Add a button for the user to click to submit text
    submit_button = st.button('Submit')

    if file is not None:
        if file.type == 'application/pdf':
            text = read_pdf(file)
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text = read_docx(file)
        else:
            st.error("Unsupported file type. Please upload a PDF or Word document.")
            return
        store_name = file.name[:-4]
    elif user_text and submit_button:
        text = user_text
        store_name = "user_text"
    else:
        st.error("Please upload a file or paste text.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)

    st.write(f'{store_name}')

    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

    query = st.text_input("Ask question's about your document:")

    suggestions = ["", "What is the main topic of the document?", "Summarize the document in 200 words?", "Provide a bullet point list of the key points mentioned in the document?", "Create the headings and subheadings for Powerpoint slides", "Translate the first paragraph to French"]

    suggestion = st.selectbox("Or select a suggestion: (ENSURE QUESTION FIELD ABOVE IS BLANK)", suggestions, index=0)

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', max_tokens=2000, temperature=0.5)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb, st.spinner('Working on response...'):
            time.sleep(3)
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)

    elif suggestion:
        query = suggestion
        docs = VectorStore.similarity_search(query=query, k=3)
        llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', max_tokens=2000, temperature=0.5)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb, st.spinner('Working on response...'):
            time.sleep(3)
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)

if __name__ == '__main__':
    main()
