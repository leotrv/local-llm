from llama_cpp import Llama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import box
import yaml


# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

llm = Llama(model_path="/home/leotraven/Development/llms/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q2_K.gguf", chat_format="llama-2")

st.title("Basic Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db3 = Chroma(persist_directory=cfg.CHROMA_PATH, embedding_function=embeddings)
    docs = db3.similarity_search(prompt, k=cfg.VECTOR_COUNT)

    context = "\n\n".join(doc.page_content for doc in docs)

    system_message = [{"role": "system", "content": f"You are an assistant who answers questions about documents. Be concise and do not use more than 3 sentences. You are given a context: {context}"}]
    messages = system_message + [{"role": message.get("role"), "content": message.get("content")} for message in st.session_state.messages]
    print(messages)

    for response in  llm.create_chat_completion(
        messages = messages,
        stream=True
        ):
            print(response)
            full_response += (response.get("choices")[0].get("delta").get("content") or "")
            message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})