import streamlit as st
from rag_backend import answer_question, get_policy_files

st.set_page_config(page_title="Zyro HR Policy Assistant", page_icon="📘")
st.title("📘 Zyro HR Policy Assistant")
st.caption("Ask questions about the available HR policy documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    policies = get_policy_files()
    if not policies:
        st.warning("Add your approved HR policy PDF files to the data/ folder, then restart.")
        st.stop()

    with st.sidebar:
        st.subheader("Available policies")
        for name in policies:
            st.caption("📄 " + name)
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("View sources"):
                    for source in msg["sources"]:
                        st.write("• " + source)

    question = st.chat_input("Ask an HR policy question...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching policy documents..."):
                try:
                    result = answer_question(question)
                    answer = result["answer"]
                    sources = result["sources"]
                    st.markdown(answer)
                    if sources:
                        with st.expander("View sources"):
                            for source in sources:
                                st.write("• " + source)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as exc:
                    st.error("Unable to process the question.")
                    st.exception(exc)
except Exception as exc:
    st.error("Application initialization failed.")
    st.exception(exc)
