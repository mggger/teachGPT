import streamlit as st
from typing import Any, Union

from graphrag.query.llm.base import BaseLLMCallback


class StreamlitLLMCallback(BaseLLMCallback):
    def __init__(self):
        super().__init__()
        self.container = st.empty()
        self.text = ""

    def on_llm_new_token(self, token: str):
        """Handle when a new token is generated."""
        super().on_llm_new_token(token)
        self.text += token
        self.container.markdown(self.text)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        """Called when LLM starts running."""
        st.write("AI is generating a response...")

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        """Called when LLM ends running."""
        st.write("AI response complete.")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Called when LLM errors."""
        st.error(f"An error occurred: {str(error)}")