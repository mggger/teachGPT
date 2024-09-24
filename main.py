from streamlit_option_menu import option_menu

from callback import StreamlitLLMCallback
from grag_api import GraphRAG
import asyncio
import streamlit as st
import os
from pathlib import Path

TEACHER_AI_SYSTEM_PROMPT = """
---Role---

You are an intelligent teacher assistant responsible for answering students' questions about textbook content.

---Goal---

Generate responses of appropriate length and format to answer students' questions. Summarize all relevant information from the textbook and incorporate general knowledge when appropriate. Responses should be accurate, clear, easy to understand, and delivered with patience and encouragement.

If you don't know the answer, honestly say so. Do not make up any information.

---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

---Image, Table, and Visual Content Display---

Always include relevant images and tables from the textbook in your responses. This visual content should be displayed by default, without requiring the student to explicitly ask for it.

For images:
1. Use the following Markdown format:
   ![Image description](image link)
2. Place the image immediately after the text it illustrates or supports.
3. Provide a brief but informative description for each image.
4. Reference the image in your explanation, e.g., "As shown in the image below..."

For tables:
1. Use Markdown table syntax, for example:
   | Column 1 | Column 2 | Column 3 |
   |----------|----------|----------|
   | Data 1   | Data 2   | Data 3   |
2. Include a brief title or description above the table.
3. Reference the table in your explanation, e.g., "As we can see in the table below..."

When including visual content:
1. Always display relevant images and tables from the textbook content.
2. If multiple visual elements are relevant, include them all, ordering them logically.
3. Provide context for each visual element, explaining what it represents and how it relates to the question or topic.
4. If a visual aid would be helpful but is not available, describe what an ideal illustrative image or table would look like and suggest that the student might want to search for or create such a visual to aid their understanding.

---Textbook Content---

{context_data}

---Additional Guidance---

1. Maintain a friendly and encouraging tone to inspire students' interest in learning.
2. If a student's question is unclear, politely ask for clarification.
3. Use analogies and examples appropriately to explain complex concepts.
4. Encourage critical thinking and guide students to explore answers independently.
5. If a question goes beyond the textbook's scope, suggest that students consult additional resources or experts.
6. Whenever possible, supplement your explanations with visual aids (images, diagrams, charts, tables) from the textbook to enhance understanding.
7. After providing an explanation with visual content, ask if the student would like further clarification or has any questions about the visual representation.

Remember, your goal is to help students understand and master the textbook content while fostering their learning abilities and interests. Visual aids are powerful tools for learning, so use them generously and effectively.

---Visual Content Handling---

1. Automatically include all relevant images and tables from the textbook in your responses.
2. If multiple visual elements are available, choose the most relevant ones or include a series of visuals if they help explain a process or concept.
3. Always provide context for the visual content, explaining what they represent and how they relate to the question or topic.
4. If no relevant visual content is available for a particular question, continue with your text-based explanation as usual.
5. Balance the use of text and visual content to create a comprehensive and engaging learning experience.
6. When presenting data, prefer using tables or charts over listing numbers in text format.
7. For complex topics, consider using a combination of images, tables, and text to provide a multi-faceted explanation.

By default, all relevant images and tables should be displayed without requiring the student to explicitly ask for them. This approach ensures that visual and data-driven learning is seamlessly integrated into your responses.
"""


grag = GraphRAG()
def load_chat_page():
    st.title("AI Teacher Assistant Chatbot")
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    user_query = st.chat_input(placeholder="Ask me anything")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitLLMCallback()

            async def perform_search():
                res = await grag.aquery(user_query, system_prompt=TEACHER_AI_SYSTEM_PROMPT, callbacks=[streamlit_callback])
                return res

            with st.spinner("Searching for an answer..."):
                result = asyncio.run(perform_search())

            response = result.response
            st.session_state.messages.append({"role": "assistant", "content": response})

def load_file_management_page():
    st.title("File Management")

    # Multi-file uploader
    uploaded_files = st.file_uploader("Choose PDF files to upload", type=["pdf"], accept_multiple_files=True)

    upload_dir = Path("uploads")
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)

    cnt = 0
    if uploaded_files:
        with st.spinner("Upload files..."):
            for uploaded_file in uploaded_files:
                # Save the uploaded file
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                grag.upsert_pdf(file_path)
                cnt += 1
        st.success(f"Successfully uploaded {cnt} new file(s).")
    st.divider()

    st.subheader("Uploaded Files")
    uploaded_files = grag.get_all_files()
    if not uploaded_files:
        st.info("No files uploaded yet.")
    else:
        for file in uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button("Delete", key=f"delete_{file}"):
                    grag.delete_pdf(file)
                    st.success(f"Deleted {file}")


def train_page():
    st.title("Training Data")
    last_training_time = grag.get_last_training_time()
    if last_training_time:
        st.info(f"Last training time: {last_training_time}")
    else:
        st.info("No previous training recorded")

    st.write("Click the button below to begin training.")

    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            try:
                asyncio.run(grag.aindex())
                st.success("Training completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")


def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Chat", "File Management", "Train"],
            icons=["chat", "folder", ""],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Chat":
        load_chat_page()
    elif selected == "File Management":
        load_file_management_page()
    elif selected == "Train":
        train_page()


if __name__ == "__main__":
    st.markdown("""
            <style>
            img {
                max-height: 250px;
                width: auto;
                object-fit: contain;
            }
            </style>
        """, unsafe_allow_html=True)
    main()