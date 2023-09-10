import streamlit as st
#from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
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

def main():
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.header("Chat to a PDF 💬👨🏻‍💻🤖")
    
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF (just one for now)", type=['pdf', 'txt', 'ppt', 'html', 'jpg', 'png'])
 
    if pdf is not None:
        # Save the uploaded file temporarily
        with open("temp_file", "wb") as f:
            f.write(pdf.getbuffer())

        # Load the file using UnstructuredFileLoader
        loader = UnstructuredFileLoader("temp_file")
        docs = loader.load()

        # Concatenate the content of all pages
        text = "".join([doc.page_content for doc in docs])

        # Remove the temporary file
        os.remove("temp_file")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

        query = st.text_input("Ask question's about your PDF file:")

        suggestions = ["", "What is the main topic of the document?", "Summarize the document in 200 words?", "Provide a bullet point list of the key points mentioned in the document?", "Create the content for Powerpoint slides based on the document content", "Translate the first paragraph to French"]
    
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