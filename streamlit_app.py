import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.title("NCERT Q&A Helper")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_box" not in st.session_state:
    st.session_state.input_box = ""

def add_message(sender, message):
    st.session_state.messages.append({"sender": sender, "message": message})

def submit_question():
    question = st.session_state.input_box.strip()
    if question:
        add_message("user", question)

        # Send to backend
        response = requests.post(f"{BACKEND_URL}/ask", json={"question": question, "session_id": "user123"})
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer received.")
            docs = data.get("retrieved_documents", [])

            # Save raw content to session state for optional citation display
            st.session_state.last_answer = answer
            st.session_state.last_docs = docs

            # Display only the answer for now
            add_message("assistant", answer)

        else:
            add_message("assistant", f"Error: {response.status_code} - {response.text}")

        # Clear input box
        st.session_state.input_box = ""

# Display chat messages
for msg in st.session_state.messages:
    if msg["sender"] == "user":
        st.markdown(
            f"<div style='text-align: right; background-color:#DCF8C6; padding:8px; margin:5px; border-radius:10px; max-width: 70%; float: right;'>{msg['message']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align: left; background-color:#E8E8E8; padding:8px; margin:5px; border-radius:10px; max-width: 70%; float: left;'>{msg['message']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='clear: both;'></div>", unsafe_allow_html=True)

# Input box with on_change callback to submit on Enter key
st.text_input("Enter your question:", key="input_box", on_change=submit_question)

if st.button("📚 Cite Sources"):

    docs = st.session_state.get("last_docs", [])
    if not docs:
        st.info("No relevant sources found.")
    else:
        st.markdown("**Retrieved Documents:**")
        for doc in docs:
            st.markdown(f"""
- **Page {doc['page']}** - [Link]({doc['link']})
  > {doc['snippet']}
""")
# Sidebar info
# st.sidebar.header("About")
# st.sidebar.info(
#     "This is a NCERT Q&A Helper. Enter your question in the text box and press Enter to get an answer "
#     "based on the available documents."
# )
