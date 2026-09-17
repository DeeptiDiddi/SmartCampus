import sqlite3
import uuid
from datetime import datetime

import streamlit as st
import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Smart Campus Assistant",
    page_icon="🎓",
    layout="wide"
)


# ==================================================
# DATABASE
# ==================================================

DB_NAME = "chat_history.db"


# ==================================================
# CREATE / MIGRATE DATABASE
# ==================================================

def create_history_table():

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    # ----------------------------------------------
    # Create original table if it doesn't exist
    # ----------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()


    # ----------------------------------------------
    # Check whether chat_id exists
    # ----------------------------------------------

    cursor.execute(
        "PRAGMA table_info(chat_history)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]


    # ----------------------------------------------
    # Add chat_id to old database
    # ----------------------------------------------

    if "chat_id" not in columns:

        cursor.execute("""
            ALTER TABLE chat_history
            ADD COLUMN chat_id TEXT
        """)

        # Existing old chats become one conversation
        cursor.execute(
            "SELECT id FROM chat_history ORDER BY id ASC"
        )

        old_rows = cursor.fetchall()


        if old_rows:

            legacy_chat_id = str(
                uuid.uuid4()
            )

            cursor.execute(
                """
                UPDATE chat_history
                SET chat_id = ?
                WHERE chat_id IS NULL
                """,
                (legacy_chat_id,)
            )


    connection.commit()
    connection.close()


# Create / migrate database
create_history_table()


# ==================================================
# CHAT FUNCTIONS
# ==================================================

def generate_chat_id():

    return str(uuid.uuid4())


# ==================================================
# CHAT TITLE
# ==================================================

def make_chat_title(question):

    title = " ".join(
        question.strip().split()
    )

    # Keep sidebar clean
    if len(title) > 60:

        title = title[:60] + "..."

    return title


# ==================================================
# SAVE CHAT MESSAGE
# ==================================================

def save_chat(
    chat_id,
    question,
    answer,
    sources
):

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO chat_history
        (
            chat_id,
            question,
            answer,
            sources,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (

        chat_id,

        question,

        answer,

        ", ".join(sources),

        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    ))

    connection.commit()
    connection.close()


# ==================================================
# GET ALL CONVERSATIONS
# ==================================================

def get_conversations():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            chat_id,
            question,
            created_at
        FROM chat_history
        WHERE id IN (
            SELECT MIN(id)
            FROM chat_history
            GROUP BY chat_id
        )
        ORDER BY id DESC
    """)

    conversations = cursor.fetchall()

    connection.close()

    return conversations


# ==================================================
# GET MESSAGES OF ONE CONVERSATION
# ==================================================

def get_chat_messages(chat_id):

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            question,
            answer,
            sources,
            created_at
        FROM chat_history
        WHERE chat_id = ?
        ORDER BY id ASC
    """, (chat_id,))

    messages = cursor.fetchall()

    connection.close()

    return messages


# ==================================================
# DELETE ONE ENTIRE CONVERSATION
# ==================================================

def delete_conversation(chat_id):

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM chat_history
        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    connection.commit()
    connection.close()


# ==================================================
# CLEAR ALL HISTORY
# ==================================================

def clear_history():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM chat_history"
    )

    connection.commit()
    connection.close()


# ==================================================
# SESSION STATE
# ==================================================

if "current_chat_id" not in st.session_state:

    st.session_state.current_chat_id = (
        generate_chat_id()
    )


if "page" not in st.session_state:

    st.session_state.page = (
        "🏠 Dashboard"
    )


# ==================================================
# LOAD RAG COMPONENTS
# ==================================================

@st.cache_resource
def load_rag():

    # ----------------------------------------------
    # ChromaDB
    # ----------------------------------------------

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = chroma_client.get_collection(
        "capstone_knowledge_base"
    )


    # ----------------------------------------------
    # Vector Store
    # ----------------------------------------------

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )


    # ----------------------------------------------
    # Embedding Model
    # ----------------------------------------------

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # ----------------------------------------------
    # Vector Index
    # ----------------------------------------------

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )


    # ----------------------------------------------
    # Retriever
    # ----------------------------------------------

    retriever = index.as_retriever(
        similarity_top_k=2
    )


    # ----------------------------------------------
    # Ollama
    # ----------------------------------------------

    llm = Ollama(
        model="qwen3.5:2b",
        request_timeout=300.0
    )


    return collection, retriever, llm


collection, retriever, llm = load_rag()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "🎓 Smart Campus"
)


# ==================================================
# NEW CHAT BUTTON
# ==================================================

if st.sidebar.button(
    "🆕 New Chat",
    use_container_width=True
):

    # Generate completely new conversation
    st.session_state.current_chat_id = (
        generate_chat_id()
    )

    # Open Ask page
    st.session_state.page = (
        "💬 Ask Smart Campus"
    )

    st.rerun()


st.sidebar.markdown("---")


# ==================================================
# CHAT HISTORY IN SIDEBAR
# ==================================================

st.sidebar.markdown(
    "### 💬 Conversations"
)


conversations = get_conversations()


if not conversations:

    st.sidebar.caption(
        "No conversations yet."
    )


else:

    for conversation in conversations:

        chat_id = conversation[0]

        first_question = conversation[1]


        # ------------------------------------------
        # Title comes from FIRST question
        # ------------------------------------------

        title = make_chat_title(
            first_question
        )


        # ------------------------------------------
        # Conversation button
        # ------------------------------------------

        if st.sidebar.button(
            f"💬 {title}",
            key=f"conversation_{chat_id}",
            use_container_width=True
        ):

            st.session_state.current_chat_id = (
                chat_id
            )

            st.session_state.page = (
                "💬 Ask Smart Campus"
            )

            st.rerun()


st.sidebar.markdown("---")


# ==================================================
# NAVIGATION
# ==================================================

st.sidebar.markdown(
    "### Navigation"
)


page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "💬 Ask Smart Campus",
        "📜 History"
    ],
    key="page"
)


st.sidebar.markdown("---")


# ==================================================
# KNOWLEDGE BASE
# ==================================================

st.sidebar.write(
    "### Knowledge Base"
)


st.sidebar.write(
    "📄 Documents: 3"
)


st.sidebar.write(
    f"🧩 Chunks: {collection.count()}"
)


st.sidebar.write(
    "🤖 Model: Qwen 3.5:2B"
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.title(
        "🎓 Smart Campus Assistant"
    )

    st.subheader(
        "College Knowledge Base Dashboard"
    )

    st.write(
        "Ask questions about academic regulations, "
        "examination rules and placement policies."
    )

    st.markdown("---")


    # ----------------------------------------------
    # Dashboard Metrics
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    conversations = get_conversations()


    total_messages = 0

    for conversation in conversations:

        chat_id = conversation[0]

        messages = get_chat_messages(
            chat_id
        )

        total_messages += len(messages)


    with col1:

        st.metric(
            "📄 Documents",
            "3"
        )


    with col2:

        st.metric(
            "🧩 Chunks",
            collection.count()
        )


    with col3:

        st.metric(
            "💬 Questions",
            total_messages
        )


    with col4:

        st.metric(
            "🤖 LLM",
            "Qwen 3.5:2B"
        )


    st.markdown("---")


    # ----------------------------------------------
    # Knowledge Base
    # ----------------------------------------------

    st.subheader(
        "📚 Knowledge Base"
    )


    st.info(
        """
Academic Regulations

Examination Rules

Placement Policy
        """
    )


    # ----------------------------------------------
    # RAG Pipeline
    # ----------------------------------------------

    st.subheader(
        "⚙️ RAG Pipeline"
    )


    st.write(
        """
Documents → Chunking → Embeddings → ChromaDB
→ Retrieval → Ollama → Answer
        """
    )


# ==================================================
# ASK SMART CAMPUS
# ==================================================

elif page == "💬 Ask Smart Campus":

    st.title(
        "💬 Ask Smart Campus"
    )


    st.write(
        "Ask a question about the college rules and policies."
    )


    # ==================================================
    # CURRENT CONVERSATION
    # ==================================================

    current_chat_id = (
        st.session_state.current_chat_id
    )


    # ==================================================
    # LOAD ONLY CURRENT CHAT
    # ==================================================

    previous_chats = get_chat_messages(
        current_chat_id
    )


    # ==================================================
    # DISPLAY PREVIOUS MESSAGES
    # ==================================================

    for item in previous_chats:

        message_id = item[0]

        old_question = item[1]

        old_answer = item[2]

        old_sources = item[3]


        # ------------------------------------------
        # User message
        # ------------------------------------------

        with st.chat_message("user"):

            st.write(
                old_question
            )


        # ------------------------------------------
        # Assistant message
        # ------------------------------------------

        with st.chat_message("assistant"):

            st.write(
                old_answer
            )


            if old_sources:

                st.markdown(
                    "### 📚 Sources"
                )


                source_list = (
                    old_sources.split(",")
                )


                for source in source_list:

                    st.write(
                        f"• {source.strip()}"
                    )


    # ==================================================
    # CHAT INPUT
    # ==================================================

    question = st.chat_input(
        "Ask your question..."
    )


    if question:

        # ==================================================
        # DISPLAY USER QUESTION
        # ==================================================

        with st.chat_message("user"):

            st.write(
                question
            )


        # ==================================================
        # CREATE CONVERSATION-AWARE RETRIEVAL QUERY
        # ==================================================

        retrieval_query = question


        # If previous messages exist in this conversation,
        # include recent questions as context.

        if previous_chats:

            recent_questions = []

            for item in previous_chats[-3:]:

                recent_questions.append(
                    item[1]
                )

            retrieval_query = "\n".join(
                recent_questions
            ) + "\n" + question


        # ==================================================
        # RETRIEVE CHUNKS
        # ==================================================

        results = retriever.retrieve(
            retrieval_query
        )


        # ==================================================
        # FILTER RELEVANT RESULTS
        # ==================================================

        filtered_results = [

            result

            for result in results

            if result.score >= 0.35

        ]


        # ==================================================
        # NO RELEVANT INFORMATION
        # ==================================================

        if not filtered_results:

            answer = (
                "I don't have enough information "
                "to answer that."
            )

            sources = []


        # ==================================================
        # RELEVANT INFORMATION FOUND
        # ==================================================

        else:

            # ------------------------------------------
            # Create Context
            # ------------------------------------------

            context = ""

            sources = []


            for result in filtered_results:

                source = (
                    result.node.metadata.get(
                        "file_name"
                    )
                )


                if source not in sources:

                    sources.append(
                        source
                    )


                context += f"""

Source: {source}

Content:
{result.node.text}

"""


            # ------------------------------------------
            # Previous conversation context
            # ------------------------------------------

            conversation_context = ""


            for item in previous_chats[-4:]:

                conversation_context += f"""

            Student:
            {item[1]}

            Assistant:
            {item[2]}

            """    
            # ==================================================
            # CONVERSATION CONTEXT
            # ==================================================

            conversation_context = ""

            if previous_chats:

                for item in previous_chats[-5:]:

                    conversation_context += f"""
            Student: {item[1]}
            Assistant: {item[2]}
            """


            # ------------------------------------------
            # Grounded Prompt
            # ------------------------------------------

            prompt = f"""
You are Smart Campus Assistant.

Answer the student's question ONLY using the information
provided in the retrieved context.

Use the conversation history to understand follow-up
questions and pronouns such as:

"it"
"that"
"they"
"this"

Do not use outside knowledge.

If the answer is not available in the retrieved context,
say exactly:

"I don't have enough information to answer that."

Give a clear and concise answer.

Conversation History:
{conversation_context}

Retrieved Context:
{context}

Current Student Question:
{question}

Answer:
"""


            # ------------------------------------------
            # Generate Answer
            # ------------------------------------------

            with st.spinner(
                "Thinking..."
            ):

                try:

                    response = llm.complete(
                        prompt
                    )

                    answer = str(
                        response
                    )


                except Exception:

                    answer = (
                        "I don't have enough information "
                        "to answer that."
                    )


        # ==================================================
        # DISPLAY ASSISTANT ANSWER
        # ==================================================

        with st.chat_message("assistant"):

            st.write(
                answer
            )


            # ------------------------------------------
            # Sources
            # ------------------------------------------

            if sources:

                st.markdown(
                    "### 📚 Sources"
                )


                for source in sources:

                    st.write(
                        f"• {source}"
                    )


            else:

                st.info(
                    "No relevant source found."
                )


        # ==================================================
        # SAVE MESSAGE
        # ==================================================

        save_chat(
            current_chat_id,
            question,
            answer,
            sources
        )


        # ------------------------------------------
        # Rerun
        # ------------------------------------------
        # This reloads the sidebar so the FIRST
        # question immediately becomes the title.
        # ------------------------------------------

        st.rerun()


# ==================================================
# HISTORY
# ==================================================

elif page == "📜 History":

    st.title(
        "📜 Chat History"
    )


    st.write(
        "Your previous Smart Campus conversations."
    )


    conversations = get_conversations()


    # ==================================================
    # NO HISTORY
    # ==================================================

    if not conversations:

        st.info(
            "No chat history yet."
        )


    # ==================================================
    # HISTORY EXISTS
    # ==================================================

    else:

        # ==================================================
        # CLEAR ALL HISTORY
        # ==================================================

        if st.button(
            "🗑️ Clear All History",
            type="primary"
        ):

            clear_history()

            # Create a fresh empty chat
            st.session_state.current_chat_id = (
                generate_chat_id()
            )

            st.success(
                "All chat history cleared."
            )

            st.rerun()


        st.markdown("---")


        # ==================================================
        # DISPLAY CONVERSATIONS
        # ==================================================

        for conversation in conversations:

            chat_id = conversation[0]

            first_question = conversation[1]

            created_at = conversation[2]


            title = make_chat_title(
                first_question
            )


            # ------------------------------------------
            # Get all messages
            # ------------------------------------------

            messages = get_chat_messages(
                chat_id
            )


            # ------------------------------------------
            # Conversation Expander
            # ------------------------------------------

            with st.expander(
                f"💬 {title}"
            ):

                col1, col2 = st.columns(
                    [4, 1]
                )


                with col1:

                    st.caption(
                        f"🕒 {created_at}"
                    )


                with col2:

                    if st.button(
                        "🗑️ Delete Chat",
                        key=f"delete_chat_{chat_id}"
                    ):

                        delete_conversation(
                            chat_id
                        )


                        # If currently open chat
                        if (
                            st.session_state.current_chat_id
                            == chat_id
                        ):

                            st.session_state.current_chat_id = (
                                generate_chat_id()
                            )


                        st.rerun()


                st.markdown("---")


                # ------------------------------------------
                # Display all messages
                # ------------------------------------------

                for item in messages:

                    question = item[1]

                    answer = item[2]

                    sources = item[3]


                    st.markdown(
                        "**👤 Student:**"
                    )

                    st.write(
                        question
                    )


                    st.markdown(
                        "**🤖 Smart Campus:**"
                    )

                    st.write(
                        answer
                    )


                    if sources:

                        st.markdown(
                            "**📚 Sources:**"
                        )


                        for source in sources.split(","):

                            st.write(
                                f"• {source.strip()}"
                            )
                    st.markdown("---")
print("hello smart campus")                   