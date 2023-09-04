'''
requirements.txt file contents:
 
langchain==0.0.154
PyPDF2==3.0.1
python-dotenv==1.0.0
streamlit==1.18.1
faiss-cpu==1.7.4
streamlit-extras
'''
import streamlit as st
#from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
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

openai.api_key = st.secrets["OPENAI_API_KEY"]
 
# Sidebar contents
with st.sidebar:
    st.title('💬 PDF Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - Streamlit
    - LangChain
    - OpenAI LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Jez')
 
##load_dotenv()
 
def main():
    st.header("Jezza's Chat with a PDF 💬👾🤖")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF (just one for now)", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl", "rb") as f:
        #         VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        #else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        suggestions = ["What is the main topic of the document?", "Can you summarize the document in 200 words?", "What are the key points mentioned in the document?", "What is the conclusion of the document?", "Can you provide an overview of the document's content?"]
        suggestion = st.selectbox("Or select a suggestion:", suggestions)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', max_tokens=2000, temperature=0.5)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

        elif suggestion:
            query = suggestion
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', max_tokens=2000, temperature=0.5)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

        else:
            response = "Please enter a query or select a suggestion to get a response."
            st.write(response)

if __name__ == '__main__':
    main()
