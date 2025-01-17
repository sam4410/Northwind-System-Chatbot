import os
import requests
import streamlit as st

CHATBOT_URL = os.getenv(
    "CHATBOT_URL", "http://127.0.0.1:8000/northwind-rag-agent"
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        LangChain agent designed to answer questions about the Northwind's customers, suppliers,
        orders, products, and insurance payers in Northwind organization.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    st.header("Example Questions")
    st.markdown("""- Who are the suppliers supplying products in "Produce" category?""")
    st.markdown(
        """- How many customers in Germany have written reviews?"""
    )
    st.markdown(
        """- What are the product categories provided by each supplier?"""
    )
    st.markdown(
        """- Which customer(s) has ordered orders with more than 5 products in it?"""
    )
    st.markdown(
        """-  Find total quantity per customer in the "Produce" category in year 2012?"""
    )
    st.markdown(
        """- What is the net sales revenue in year 2012?"""
    )
    st.markdown(
        """- Which country had the largest percent increase in number of orders
        from 2012 to 2013?"""
    )

st.title("Northwind Internal Chatbot")
st.info(
    """Ask me questions about orders, customers, products, categories,
    suppliers and reviews!"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["output"]
            explanation = response.json()["intermediate_steps"]

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated?", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )
