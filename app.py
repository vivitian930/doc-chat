from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def create_db_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)


def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    with get_openai_callback() as cb:
        response = chain.run(question=query, docs=docs)
        print(cb)

    # response = response.replace("\n", "")
    return response, docs


def main():
    load_dotenv()
    # Initialise session state variables
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "reference" not in st.session_state:
        st.session_state["reference"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "chat" not in st.session_state:
        st.session_state["chat"] = False

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your docs 💬")

    st.sidebar.title("Load your PDF")
    # counter_placeholder = st.sidebar.empty()
    # counter_placeholder.write(f"Upload your PDF Here 👇")

    pdf = st.sidebar.file_uploader("Upload your PDF", type="pdf")
    # extract the text
    if pdf is not None:
        db = create_db_from_pdf(pdf)
        st.sidebar.success("Successfully generated indexes.")
        st.session_state["chat"] = True

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    if st.session_state["chat"]:
        with container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_area("Ask a question:", key="input", height=100)
                submit_button = st.form_submit_button(label="Submit")

            if submit_button and user_input:
                response, docs = get_response_from_query(db, query=user_input, k=3)
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(response)
                st.session_state["reference"].append(docs)

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                st.markdown("__USER QUESTION__")
                # st.message(
                #     st.session_state["past"][i], is_user=True, key=str(i) + "_user"
                # )
                st.write(st.session_state["past"][i])
                st.divider()
                st.markdown("__ANSWER__")
                # st.message(st.session_state["generated"][i], key=str(i))
                st.write(st.session_state["generated"][i])
                for idx, doc in enumerate(st.session_state["reference"][i]):
                    with st.expander(f"Source {idx}"):
                        st.write(doc.page_content)
                st.divider()


if __name__ == "__main__":
    main()
