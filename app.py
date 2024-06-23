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
import fitz  # PyMuPDF
import base64
import io
from PIL import Image

openai.api_key = st.secrets["OPENAI_API_KEY"]

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

def read_pdf(file):
    pdf_file = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    images = []
    for page_num in range(len(pdf_file)):
        page = pdf_file[page_num]
        text += page.get_text()
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_str)
    return text, images

def read_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

def run_conversation(query, text, images):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Analyze both the text and images provided to answer the user's question."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Text content: {text}\n\nQuestion: {query}"},
                *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in images[:5]]  # Limit to 5 images
            ]
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=4000,
    )
    
    return response.choices[0].message.content

def main():
    if 'text' not in st.session_state:
        st.session_state['text'] = ""
    if 'images' not in st.session_state:
        st.session_state['images'] = []

    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.header("Chat to a Document 💬👨🏻‍💻🤖")
    
    input_method = st.radio("Choose your input method:", ("Upload a document", "Paste text or web address"))

    if input_method == "Upload a document":
        file = st.file_uploader("Load your PDF or Word document (just one for now)", type=['pdf', 'docx'])

        if file is not None:
            if file.type == 'application/pdf':
                st.session_state['text'], st.session_state['images'] = read_pdf(file)
            elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                st.session_state['text'] = read_docx(file)
                st.session_state['images'] = []  # No images for docx in this implementation
            else:
                st.error("Unsupported file type. Please upload a PDF or Word document.")
                return
        else:
            st.error("Please upload a file.")
            return
    else:
        text_or_url = st.text_area("Paste your text or URL here: URLS must be in format https://")
        process_button = st.button("Process Text")
        
        if process_button:
            if text_or_url:
                if text_or_url.startswith('http://') or text_or_url.startswith('https://'):
                    response = requests.get(text_or_url)
                    if 'text/html' in response.headers['Content-Type']:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        text = ' '.join(p.get_text() for p in soup.find_all('p'))
                        st.text_area("**Fetched Information from site:  Note that some websites block content access so the fetched information may be limited**", text)
                    else:
                        text = response.text
                        st.text_area("Fetched Information:", text)
                else:
                    text = text_or_url
                st.session_state['text'] = text
                st.session_state['images'] = []  # No images for pasted text or URLs in this implementation

    if not st.session_state['text']:
        st.error("Please provide some text either by uploading a document or pasting text.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=st.session_state['text'])

    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    query = st.text_input("Ask question's about your document:")

    suggestions = ["", "What is the main topic of the document?", "Summarize the document in 200 words?", "Provide a bullet point list of the key points mentioned in the document?", "Create the headings and subheadings for Powerpoint slides", "Translate the first paragraph to French"]

    suggestion = st.selectbox("Or select a suggestion: (ENSURE QUESTION FIELD ABOVE IS BLANK)", suggestions, index=0)

    if query or suggestion:
        if suggestion:
            query = suggestion

        with st.spinner('Working on response...'):
            time.sleep(3)
            response = run_conversation(query, st.session_state['text'], st.session_state['images'])
        st.write(response)

if __name__ == '__main__':
    main()
