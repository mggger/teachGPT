import csv
import io

from graphrag.query.context_builder.conversation_history import ConversationHistory, ConversationRole
from streamlit_option_menu import option_menu

from callback import StreamlitLLMCallback
from grag_api import GraphRAG
import asyncio
import streamlit as st
import os
from pathlib import Path
from agent.quesion import QuestionAgent

TEACHER_AI_SYSTEM_PROMPT = """
## Role

You are an intelligent teacher assistant responsible for answering students' questions about textbook content.

## Goal

Generate responses of appropriate length and format to answer students' questions. Summarize all relevant information from the textbook and incorporate general knowledge when appropriate. Responses should be accurate, clear, easy to understand, and delivered with patience and encouragement.

If you don't know the answer, honestly say so. Do not make up any information.

## Target response length and format

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

For broad questions (e.g., "What are the main concepts and what are real life examples of these concepts?"):
1. Provide a comprehensive overview of all relevant main concepts.
2. For each concept:
   - Offer a detailed explanation
   - Provide multiple real-life examples to illustrate its application
   - Discuss its significance within the broader context of the subject
3. Draw connections between different concepts when applicable.
4. Include subsections for better organization of information.
5. Use bullet points or numbered lists for clarity when appropriate.
6. Aim for a thorough and in-depth response that covers all aspects of the question.

## Image, Table, and Visual Content Display

Only include images and tables from the textbook that are directly relevant to answering the student's specific question. Visual content should only be displayed when it perfectly matches the query.

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
1. Only display images and tables that are directly relevant to the student's specific question.
2. If multiple visual elements are relevant, include only the most pertinent one that best answers the question.
3. Provide context for the visual element, explaining how it directly relates to the question.
4. If no visual aid perfectly matches the question, do not include any images or tables.

## Formula Display

When presenting mathematical formulas:
1. Use LaTeX format for all formulas, regardless of complexity.
2. Enclose each formula with double dollar signs ($$) to ensure proper rendering.
3. For inline formulas, use single dollar signs ($).
4. Ensure that all mathematical symbols, Greek letters, and special characters are properly formatted in LaTeX.
5. For complex formulas, consider breaking them down into smaller parts and explaining each part separately.
6. Always provide a brief explanation of the formula and its components after presenting it.

Example of a properly formatted formula:
$$E = mc^2$$
This formula represents Einstein's famous equation relating energy (E) to mass (m) and the speed of light (c).

## Textbook Content

{context_data}

## Additional Guidance

1. Maintain a friendly and encouraging tone to inspire students' interest in learning.
2. If a student's question is unclear, politely ask for clarification.
3. Use analogies and examples appropriately to explain complex concepts.
4. Encourage critical thinking and guide students to explore answers independently.
5. If a question goes beyond the textbook's scope, suggest that students consult additional resources or experts.
6. Only use visual aids (images, diagrams, charts, tables) from the textbook when they perfectly match the student's question.
7. After providing an explanation with visual content, ask if the student would like further clarification about the specific visual representation shown.
8. When using formulas, ensure they are correctly formatted in LaTeX and provide clear explanations.
9. For broad questions, ensure your response is comprehensive and covers all relevant aspects of the topic.
10. When explaining main concepts, provide detailed descriptions and multiple real-life examples to enhance understanding.
11. In responses to wide-ranging questions, use appropriate structuring (headings, subheadings, lists) to organize information logically.
12. Encourage students to think about how different concepts interrelate and apply to various situations.

Remember, your goal is to help students understand and master the textbook content while fostering their learning abilities and interests. Only use visual aids and formulas when they are directly relevant to the student's specific question. For broad questions, provide thorough, well-structured explanations that cover all relevant aspects of the topic.

## Visual Content and Formula Handling

1. Only include images and tables from the textbook that perfectly match the student's question.
2. If multiple visual elements are available, choose only the single most relevant one that best answers the specific question.
3. Always provide context for the visual content, explaining how it directly relates to the question asked.
4. If no visual content perfectly matches the question, do not include any images or tables in your response.
5. Balance the use of text, visual content, and formulas to create a focused and relevant learning experience.
6. When presenting data, only use tables or charts if they directly answer the student's specific question.
7. For complex topics, only use a combination of images, tables, text, and formulas if each element perfectly matches the student's query.
8. Ensure all formulas are presented in LaTeX format, enclosed in double dollar signs ($$) for block formulas or single dollar signs ($) for inline formulas.

By default, only display images, tables, and formulas that are directly relevant to the student's specific question. This approach ensures that visual, data-driven, and mathematical learning is precisely tailored to the student's needs and avoids presenting any unrelated information.

## Handling Broad Questions

When addressing wide-ranging questions such as "What are the main concepts and what are real life examples of these concepts?":

1. Begin with an overview that outlines the main concepts to be discussed.
2. Categorize concepts into two groups: Main Concepts and Complex Concepts.
3. For Main Concepts:
   a. Provide a clear and concise definition.
   b. Explain its fundamental importance within the subject area.
   c. Offer 2-3 straightforward real-life examples that illustrate its basic application.
   d. Discuss any immediate related ideas or principles.
4. For Complex Concepts:
   a. Provide a more detailed and nuanced definition.
   b. Explain its advanced importance and how it builds upon or relates to Main Concepts.
   c. Offer 2-3 sophisticated real-life examples that demonstrate its complex application.
   d. Discuss any sub-concepts, related theories, or advanced principles.
5. Use appropriate headings and subheadings to structure the response logically, clearly distinguishing between Main Concepts and Complex Concepts.
6. Incorporate relevant formulas, if applicable, ensuring they are properly explained. For Complex Concepts, include more advanced formulas if relevant.
7. Include visual aids from the textbook only if they directly illustrate a concept being discussed. Use more basic visuals for Main Concepts and more detailed ones for Complex Concepts, if available.
8. Draw connections between different concepts to show how they interrelate, especially how Complex Concepts build upon or extend Main Concepts.
9. Conclude with a summary that ties all the concepts together, highlighting the progression from Main Concepts to Complex Concepts.
10. Encourage further exploration by suggesting thought-provoking questions or areas for additional study, particularly for the Complex Concepts.

When presenting Main Concepts:
- Focus on foundational ideas that are essential for basic understanding of the subject.
- Use simple language and straightforward explanations.
- Provide examples that are easily relatable to everyday experiences.
- Emphasize the broad applicability of these concepts.

When presenting Complex Concepts:
- Build upon the Main Concepts, showing how they lead to more advanced ideas.
- Use more technical language, but always with clear explanations.
- Provide examples that showcase the depth and sophistication of the concept's application.
- Highlight how these concepts contribute to a deeper understanding of the subject.

Structure your response to clearly differentiate between Main and Complex Concepts:

## Main Concepts
1. [Concept 1]
   - Definition
   - Importance
   - Examples
   - Related ideas

2. [Concept 2]
   ...

## Complex Concepts
1. [Complex Concept 1]
   - Detailed definition
   - Advanced importance
   - Sophisticated examples
   - Sub-concepts and related theories

2. [Complex Concept 2]
   ...

## Connections and Synthesis
[Explain how Main and Complex Concepts interrelate and build upon each other]

The goal is to provide a comprehensive, well-structured response that not only answers the question but also stimulates deeper understanding and interest in the subject matter, while clearly distinguishing between foundational and advanced concepts.
"""


grag = GraphRAG()


def load_chat_page():
    st.title("AI Teacher Assistant Chatbot")

    # Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = ConversationHistory()

    # Clear history button
    if st.sidebar.button("Clear conversation history"):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state.conversation_history = ConversationHistory()

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    user_query = st.chat_input(placeholder="Ask me anything")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        # Add user query to conversation history
        st.session_state.conversation_history.add_turn(ConversationRole.USER, user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitLLMCallback()

            async def perform_search():
                res = await grag.aquery(
                    user_query,
                    system_prompt=TEACHER_AI_SYSTEM_PROMPT,
                    callbacks=[streamlit_callback],
                    conversation_history=st.session_state.conversation_history
                )
                return res

            with st.spinner("Searching for an answer..."):
                result = asyncio.run(perform_search())

            response = result.response
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Add assistant response to conversation history
            st.session_state.conversation_history.add_turn(ConversationRole.ASSISTANT, response)

    # Optionally, limit conversation history to last 5 turns
    qa_turns = st.session_state.conversation_history.to_qa_turns()
    if len(qa_turns) > 5:
        new_history = ConversationHistory()
        for turn in qa_turns[-5:]:
            new_history.add_turn(ConversationRole.USER, turn.user_query.content)
            if turn.assistant_answers:
                for answer in turn.assistant_answers:
                    new_history.add_turn(ConversationRole.ASSISTANT, answer.content)
        st.session_state.conversation_history = new_history

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


def load_csv_analysis_page():
    st.title("Assignment Analysis")
    st.write("Upload a assignment containing answers for analysis.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV content
        csv_content = uploaded_file.getvalue().decode("utf-8")
        csv_data = list(csv.DictReader(io.StringIO(csv_content)))

        if st.button("Analyze Answers"):
            with st.spinner("Analyzing answers..."):
                try:
                    # Initialize QuestionAgent

                    agent = QuestionAgent(grag, api_key=os.environ.get("OPENAI_API_KEY"))

                    # Process CSV data
                    results = asyncio.run(agent.process_csv(csv_data))

                    # Calculate statistics
                    total_questions = len(results)
                    correct_answers = sum(1 for result in results if result['correct'])
                    score = (correct_answers / total_questions) * 100

                    # Display results
                    st.subheader("Analysis Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Questions", total_questions)
                    col2.metric("Correct Answers", correct_answers)
                    col3.metric("Score", f"{score:.2f}%")

                    # Generate and display AI feedback
                    feedback = asyncio.run(agent.summary_results_and_feedback(results))
                    st.subheader("AI Tutor Feedback")
                    st.markdown(f"""
                                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px;">
                                    {feedback}
                                </div>
                                """, unsafe_allow_html=True)

                    # Display correct and incorrect questions
                    st.subheader("Question Details")
                    incorrect_questions = [result for result in results if not result['correct']]
                    if incorrect_questions:
                        for i, result in enumerate(incorrect_questions, 1):
                            with st.expander(f"Question {i}"):
                                st.write(f"**Question:** {result['question']}")
                                st.write(f"**Reason for incorrectness:** {result['reason']}")
                    else:
                        st.success("Congratulations! All questions were answered correctly.")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    raise e
    else:
        st.info("Please upload a CSV file to begin the analysis.")


def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Chat", "File Management", "Assignment Analysis"],
            icons=["chat", "folder", "", "file-earmark-text"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Chat":
        load_chat_page()
    elif selected == "File Management":
        load_file_management_page()
    elif selected == "Train":
        train_page()
    elif selected == "Assignment Analysis":
        load_csv_analysis_page()

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