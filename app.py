import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from anthropic import Anthropic
import os
import time
import pandas as pd
from io import BytesIO
from PIL import Image

# Set up Anthropic API key
anthropic = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

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

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def process_image(file):
    image = Image.open(file)
    
    # Resize image if it's too large
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

def display_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", use_column_width=True)

def main():
    if 'text' not in st.session_state:
        st.session_state['text'] = ""
    if 'image_ref' in st.session_state:
        del st.session_state['image_ref']

    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.header("Chat to a Document or Image 💬👨🏻‍💻🤖")
    
    input_method = st.radio("Choose your input method:", ("Upload a document or image", "Paste text or web address"))

    if input_method == "Upload a document or image":
        file = st.file_uploader("Load your PDF, Word, CSV document, or Image", type=['pdf', 'docx', 'csv', 'png', 'jpg', 'jpeg'])

        if file is not None:
            if file.type == 'application/pdf':
                st.session_state['text'] = read_pdf(file)
            elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                st.session_state['text'] = read_docx(file)
            elif file.type == 'text/csv':
                st.session_state['text'] = read_csv(file)
            elif file.type in ['image/png', 'image/jpeg']:
                image_bytes = process_image(file)
                image_response = anthropic.images.create(image=image_bytes)
                st.session_state['image_ref'] = image_response.image_id
                st.session_state['text'] = "An image has been uploaded."
                display_image(image_bytes)
            else:
                st.error("Unsupported file type. Please upload a PDF, Word, CSV document, or image.")
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

    if not st.session_state['text'] and 'image_ref' not in st.session_state:
        st.error("Please provide some text or an image either by uploading a document/image or pasting text.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=st.session_state['text'])

    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    query = st.text_input("Ask questions about your document or image:")

    suggestions = [
        "",
        "What is the main topic of the document?",
        "Summarize the document in 200 words?",
        "Provide a bullet point list of the key points mentioned in the document?",
        "Create the headings and subheadings for Powerpoint slides",
        "Translate the first paragraph to French",
        "What is the first line entry in the spreadsheet?",
        "How many rows are in the CSV file?",
        "Describe the image in detail",
        "What objects can you see in the image?",
        "What is the dominant color in the image?"
    ]

    suggestion = st.selectbox("Or select a suggestion: (ENSURE QUESTION FIELD ABOVE IS BLANK)", suggestions, index=0)

    if query or suggestion:
        if suggestion:
            query = suggestion
        docs = VectorStore.similarity_search(query=query, k=3)
        
        context = "\n".join([doc.page_content for doc in docs])
        
        message = f"""
        Human: Answer this question using the provided context and image (if available).

        Question: {query}

        Context:
        {context}

        """

        if 'image_ref' in st.session_state:
            message += f"\nAn image is attached to this message. The image reference is: {st.session_state['image_ref']}"

        message += "\nAssistant: Here's the answer based on the provided context and image (if any):"

        st.text_area("Debug: Message sent to API", message, height=300)

        with st.spinner('Working on response...'):
            response = anthropic.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4000,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": message}
                ],
                image_references=[st.session_state['image_ref']] if 'image_ref' in st.session_state else None
            )
        
        st.write(response.content[0].text)

if __name__ == '__main__':
    main()
