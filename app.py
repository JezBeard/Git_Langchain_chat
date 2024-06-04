import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
#from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
    if 'text' not in st.session_state:
        st.session_state['text'] = ""

    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.header("Chat to a Document 💬👨🏻‍💻🤖")
    
    # Add a radio button for the user to select the input method
    input_method = st.radio("Choose your input method:", ("Upload a document", "Paste text or web address"))

    if input_method == "Upload a document":
        # upload a PDF or Word file
        file = st.file_uploader("Load your PDF or Word document (just one for now)", type=['pdf', 'docx'])

        if file is not None:
            if file.type == 'application/pdf':
                st.session_state['text'] = read_pdf(file)
            elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                st.session_state['text'] = read_docx(file)
            else:
                st.error("Unsupported file type. Please upload a PDF or Word document.")
                return
        else:
            st.error("Please upload a file.")
            return
    else:
        # Paste text or URL
        text_or_url = st.text_area("Paste your text or URL here: URLS must be in format https://")
        process_button = st.button("Process Text")
        
        if process_button:
            if text_or_url:
                # Check if it's a URL
                if text_or_url.startswith('http://') or text_or_url.startswith('https://'):
                    # It's a URL, fetch the content
                    response = requests.get(text_or_url)

                    # Check if it's a HTML page
                    if 'text/html' in response.headers['Content-Type']:
                        # Parse the HTML content with BeautifulSoup
                        soup = BeautifulSoup(response.content, 'html.parser')
                        # Extract all paragraph texts
                        text = ' '.join(p.get_text() for p in soup.find_all('p'))
                        st.text_area("**Fetched Information from site:  Note that some websites block content access so the fetched information may be limited**", text)  # Display the fetched information in a text box
                    else:
                        # It's not a HTML page, just use the raw content
                        text = response.text
                        st.text_area("Fetched Information:", text)  # Display the fetched information in a text box
                else:
                    # It's not a URL, just use the pasted text
                    text = text_or_url
                st.session_state['text'] = text  # Store the text in the session state

    # Check if text is provided
    if not st.session_state['text']:  # Use the text from the session state
        st.error("Please provide some text either by uploading a document or pasting text.")
        return

    # Process the pasted text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text=st.session_state['text'])  # Use the text from the session state

    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    query = st.text_input("Ask question's about your document:")

    suggestions = ["", "What is the main topic of the document?", "Summarize the document in 200 words?", "Provide a bullet point list of the key points mentioned in the document?", "Create the headings and subheadings for Powerpoint slides", "Translate the first paragraph to French"]

    suggestion = st.selectbox("Or select a suggestion: (ENSURE QUESTION FIELD ABOVE IS BLANK)", suggestions, index=0)

    system_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-4o', max_tokens=4000, temperature=0.2)
        
        message = """
        Answer this question using the provided context only.

        {question}

        Context:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages([("human", message)])

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=VectorStore.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
        )
        
        with get_openai_callback() as cb, st.spinner('Working on response...'):
            time.sleep(3)
            response = chain.run(query)
            print(cb)
        st.write(response)

    elif suggestion:
        query = suggestion
        docs = VectorStore.similarity_search(query=query, k=3)
        llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-4o', max_tokens=4000, temperature=0.1)
        
        message = """
        Answer this question using the provided context only. 

        {question}

        Context:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages([("human", message)])

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=VectorStore.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
        )
        
        with get_openai_callback() as cb, st.spinner('Working on response...'):
            time.sleep(3)
            response = chain.run(query)
            print(cb)
        st.write(response)


if __name__ == '__main__':
    main()
