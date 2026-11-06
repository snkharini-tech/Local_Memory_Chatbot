
import os
import json
import uuid
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(
    page_title="NEXORA",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Main background */

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #101D2E !important;
    color: #F8FAFC !important;
}

header[data-testid="stHeader"] {
    background: #101D2E !important;
}

/* Sidebar */

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #152438 !important;
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}

/* SELECT BOX - IMPORTANT TEXT FIX */

section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"] {
    background-color: #20364A !important;
    color: #FFFFFF !important;
    border-color: #5EEAD4 !important;
}

section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #20364A !important;
    color: #FFFFFF !important;
    border: 1px solid #5EEAD4 !important;
}

section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    background-color: transparent;
}

section[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
    color: #5EEAD4 !important;
    fill: #5EEAD4 !important;
}

/* DROPDOWN OPTIONS */

[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #20364A !important;
}

[role="option"] {
    background-color: #20364A !important;
    color: #FFFFFF !important;
}

[role="option"] * {
    color: #FFFFFF !important;
}

[role="option"]:hover,
[aria-selected="true"] {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
}

/* MEMORY INFORMATION CARD */

.memory-card {
    background-color: #20364A !important;
    border: 1px solid #5EEAD4 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin: 10px 0 18px 0 !important;
    color: #FFFFFF !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

.memory-card * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* MAIN TITLE */

.nexora-title {
    text-align: center;
    font-size: 55px;
    font-weight: 800;
    letter-spacing: 3px;
    color: #5EEAD4 !important;
}

.nexora-subtitle {
    text-align: center;
    color: #DCE7F2 !important;
    font-size: 17px;
    margin-bottom: 25px;
}

/* STATUS BAR */

.status-line {
    text-align: center;
    color: #5EEAD4 !important;
    background: #1C3045 !important;
    border: 1px solid #347B78 !important;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 25px;
}

/* CHAT MESSAGES */

[data-testid="stChatMessage"] {
    background-color: #20364A !important;
    border: 1px solid #34516B !important;
    border-radius: 15px !important;
    padding: 15px !important;
    margin-bottom: 12px !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li {
    color: #FFFFFF !important;
}

/* CHAT INPUT */

[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: #101D2E !important;
}

[data-testid="stChatInput"] {
    background: #1C3045 !important;
    border: 1px solid #5EEAD4 !important;
    border-radius: 14px !important;
}

[data-testid="stChatInput"] textarea {
    background: #1C3045 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #CBD5E1 !important;
    -webkit-text-fill-color: #CBD5E1 !important;
}

/* BUTTONS */

.stButton > button {
    width: 100%;
    min-height: 42px;
    background: #0F766E !important;
    color: #FFFFFF !important;
    border: 1px solid #5EEAD4 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background: #115E59 !important;
}

/* EXPANDER */

[data-testid="stExpander"] {
    background: #1C3045 !important;
    border: 1px solid #52718A !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] p {
    color: #FFFFFF !important;
}

/* GENERAL TEXT */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp p,
.stApp label {
    color: #F8FAFC;
}

/* FOOTER */

hr {
    border-color: #34516B !important;
}

footer, #MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

CHAT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nexora_chats.json"
)


def save_conversations():

    data = {}

    for cid, chat in st.session_state.conversations.items():

        data[cid] = {
            "title": chat["title"],
            "messages": chat["messages"]
        }

    with open(CHAT_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_conversations():

    if not os.path.exists(CHAT_FILE):
        return {}

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            cid: {
                "title": chat.get("title", "New Conversation"),
                "messages": chat.get("messages", []),
                "memories": {}
            }
            for cid, chat in data.items()
        }

    except (OSError, json.JSONDecodeError):
        return {}
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()


def create_conversation():

    cid = str(uuid.uuid4())

    st.session_state.conversations[cid] = {
        "title": "New Conversation",
        "messages": [],
        "memories": {}
    }

    st.session_state.active_conversation_id = cid

    save_conversations()


if "active_conversation_id" not in st.session_state:

    if st.session_state.conversations:
        st.session_state.active_conversation_id = next(
            iter(st.session_state.conversations)
        )
    else:
        create_conversation()


if not st.session_state.conversations:
    create_conversation()


if st.session_state.active_conversation_id not in st.session_state.conversations:
    st.session_state.active_conversation_id = next(
        iter(st.session_state.conversations)
    )


MEMORY_TYPES = [
    "Buffer Memory",
    "Summary Memory",
    "Window Memory"
]

def get_ollama_url():

    url = os.getenv("OLLAMA_BASE_URL")

    if not url:
        try:
            url = st.secrets.get(
                "OLLAMA_BASE_URL",
                "http://localhost:11434"
            )
        except Exception:
            url = "http://localhost:11434"

    return url.rstrip("/")


@st.cache_resource
def load_llm(model_name, base_url):

    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0.7
    )

def create_memory(memory_type, llm):

    if memory_type == "Buffer Memory":

        return ConversationBufferMemory(
            memory_key="history",
            input_key="input",
            output_key="output",
            return_messages=False
        )

    elif memory_type == "Summary Memory":

        return ConversationSummaryMemory(
            llm=llm,
            memory_key="history",
            input_key="input",
            output_key="output",
            return_messages=False
        )

    elif memory_type == "Window Memory":

        return ConversationBufferWindowMemory(
            k=3,
            memory_key="history",
            input_key="input",
            output_key="output",
            return_messages=False
        )


def get_memory(chat, memory_type, llm):

    if memory_type not in chat["memories"]:

        memory = create_memory(memory_type, llm)

        messages = chat["messages"]

        for i in range(0, len(messages) - 1, 2):

            user_message = messages[i]
            assistant_message = messages[i + 1]

            if (
                user_message["role"] == "user"
                and assistant_message["role"] == "assistant"
            ):

                memory.save_context(
                    {"input": user_message["content"]},
                    {"output": assistant_message["content"]}
                )

        chat["memories"][memory_type] = memory

    return chat["memories"][memory_type]


def generate_response(user_input, llm, memory):

    history = memory.load_memory_variables({}).get(
        "history", ""
    )

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            You are NEXORA, an intelligent and friendly AI assistant.

            Instructions:
            - Give correct and useful answers.
            - Remember relevant conversation history.
            - Answer in the language requested by the user.
            - Give clear explanations.
            - Use examples when necessary.
            - Keep answers organized.
            - Avoid unnecessary repetition.
            """
        ),

        (
            "human",
            """
            Previous Conversation:

            {history}

            Current Question:

            {input}
            """
        )

    ])

    response = (prompt | llm).invoke({
        "history": history,
        "input": user_input
    })

    answer = response.content

    if not isinstance(answer, str):
        answer = str(answer)

    memory.save_context(
        {"input": user_input},
        {"output": answer}
    )

    return answer

st.sidebar.markdown(
    "<h2 style='color:#5EEAD4;'>💬 NEXORA</h2>",
    unsafe_allow_html=True
)

st.sidebar.caption("Your personal AI workspace")

if st.sidebar.button(
    "＋ New Conversation",
    use_container_width=True
):

    create_conversation()
    st.rerun()


st.sidebar.markdown("---")

st.sidebar.subheader("Conversation History")


visible_chats = [
    (cid, chat)
    for cid, chat in st.session_state.conversations.items()
    if chat["messages"]
]

st.sidebar.caption(
    f"{len(visible_chats)} saved conversation(s)"
)


for cid, saved_chat in reversed(visible_chats):

    title = saved_chat["title"]

    if len(title) > 25:
        title = title[:25] + "..."

    prefix = (
        "● " if cid == st.session_state.active_conversation_id
        else "💬 "
    )

    if st.sidebar.button(
        prefix + title,
        key=f"chat_{cid}",
        use_container_width=True
    ):

        st.session_state.active_conversation_id = cid
        st.rerun()


st.sidebar.markdown("---")

st.sidebar.subheader("NEXORA Settings")

model_name = st.sidebar.selectbox(
    "Select Local LLM",
    [
        "qwen2.5:3b",
        "gemma3:1b",
        "llama3.2",
        "mistral"
    ],
    key="selected_model"
)

memory_type = st.sidebar.selectbox(
    "Select Memory Type",
    MEMORY_TYPES,
    key="selected_memory"
)

memory_descriptions = {

    "Buffer Memory":
        "Stores the complete conversation history.",

    "Summary Memory":
        "Creates a summary of previous conversations.",

    "Window Memory":
        "Remembers the latest 3 conversation exchanges."

}


st.sidebar.markdown(
    f"""
    <div class="memory-card">
        {memory_descriptions[memory_type]}
    </div>
    """,
    unsafe_allow_html=True
)

if st.sidebar.button(
    "🗑️ Delete Current Conversation",
    use_container_width=True
):

    current_id = st.session_state.active_conversation_id

    del st.session_state.conversations[current_id]

    save_conversations()

    if st.session_state.conversations:

        st.session_state.active_conversation_id = next(
            iter(st.session_state.conversations)
        )

    else:
        create_conversation()

    st.rerun()

if st.sidebar.button(
    "Clear All Conversations",
    use_container_width=True
):

    st.session_state.conversations = {}

    create_conversation()

    save_conversations()

    st.rerun()

conversation = st.session_state.conversations[
    st.session_state.active_conversation_id
]

st.markdown(
    '<div class="nexora-title">NEXORA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="nexora-subtitle">'
    'Think clearly. Ask anything. Explore with AI.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="status-line">'
    f'Active Memory: {memory_type} '
    f'| Model: {model_name}'
    f'</div>',
    unsafe_allow_html=True
)

try:

    llm = load_llm(
        model_name,
        get_ollama_url()
    )

    memory = get_memory(
        conversation,
        memory_type,
        llm
    )

except Exception as error:

    st.error(f"Model initialization failed: {error}")

    st.info(
        "Check Ollama is running and the selected model is installed. "
        "Run: ollama list"
    )

    st.stop()

if not conversation["messages"]:

    st.markdown(
        """
        <h2 style="text-align:center;color:#5EEAD4;">
            Welcome to NEXORA 👋
        </h2>
        <p style="text-align:center;color:#DCE7F2;">
            Your personal AI assistant is ready.
            Ask me anything!
        </p>
        """,
        unsafe_allow_html=True
    )

for message in conversation["messages"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Message NEXORA...")


if user_input:

    if conversation["title"] == "New Conversation":

        conversation["title"] = (
            user_input.strip().split("\n")[0][:32]
            or "New Conversation"
        )


    with st.chat_message("user"):
        st.markdown(user_input)


    conversation["messages"].append({
        "role": "user",
        "content": user_input
    })

    save_conversations()

    with st.chat_message("assistant"):

        with st.spinner("NEXORA is thinking..."):

            try:

                answer = generate_response(
                    user_input,
                    llm,
                    memory
                )

                st.markdown(answer)

                conversation["messages"].append({
                    "role": "assistant",
                    "content": answer
                })

                save_conversations()

            except Exception as error:

                st.error(
                    f"Unable to generate response: {error}"
                )

                st.info(
                    "Check Ollama and the selected model. "
                    "Run: ollama list"
                )



with st.expander("🧠 View Memory Contents"):

    st.write("Selected Memory:", memory_type)

    st.write(
        memory.load_memory_variables({}).get(
            "history",
            "No memory available yet."
        )
    )


st.markdown("---")

st.markdown(
    """
    <p style="text-align:center;color:#CBD5E1;">
        NEXORA AI | Powered by Ollama
    </p>
    """,
    unsafe_allow_html=True
)
