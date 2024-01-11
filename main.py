import streamlit as st
from app import create_vector_db, get_qa_chain
from ui import user_template, assistant_template

def display_conversation(history):

    for i in range(len(history["assistant"])):
        st.write(user_template.replace(
                "{{MSG}}", history["user"][i]), unsafe_allow_html=True)
        st.write(assistant_template.replace(
                "{{MSG}}", history["assistant"][i]), unsafe_allow_html=True)

def vectorize_pdf():
    st.sidebar.subheader("Upload your documents")
    pdf_docs = st.sidebar.file_uploader("Upload your PDFs and click on 'Process'", accept_multiple_files=True)
    if st.sidebar.button("Process"):
        create_vector_db(pdf_docs)
        st.sidebar.write("Vector Store creation completed")
    return True


st.set_page_config(page_title="Chat with a Travel Buddy")
if "assistant" not in st.session_state:
        st.session_state["assistant"] = ["How can I help you."]
if "user" not in st.session_state:
    st.session_state["user"] = ["Hi!"]

st.title("Chat with a Travel Buddy")
flag = vectorize_pdf()
user_question = st.text_input("Ask a question:")

while flag:
    chain = get_qa_chain()
    break    

if st.button("Continue"):
    processing_placeholder = st.empty()
    processing_placeholder.text("In processing....")
    
    answer = chain({"query": user_question})["result"]

    processing_placeholder.text("Processing complete!")
    st.session_state["user"].append(user_question)
    st.session_state["assistant"].append(answer)

if st.session_state["assistant"]:
    display_conversation(st.session_state)