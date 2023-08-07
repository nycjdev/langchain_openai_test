from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS #a lib for sim search and clusing of vectors
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def main():
    print('hello world!')
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")

    #upload a file
    pdf=st.file_uploader('upload your pdf', type="pdf")

    #extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()
        #st.write(text)

        #split into chunks
        text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, #next chunk will contain 200 chars before
        length_function=len
        )
       # text = "hello world"
        chunks=text_splitter.split_text(text)
        #st.write(chunks)

        
    #create embeddings
        embeddings = OpenAIEmbeddings()
        st.write("embedding...")
        knowledge_base=FAISS.from_texts(chunks, embeddings)

    #show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            search_result=knowledge_base.similarity_search(user_question)
            #st.write(search_result)
            llm=OpenAI()
            chain=load_qa_chain(llm,chain_type="stuff")
            response = chain.run(input_documents=search_result, question=user_question)
            st.write(response)
    

if __name__=='__main__':
    print('name: '+__name__)
    main()

