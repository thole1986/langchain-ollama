### Question Answering using LLM
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler 
from langchain_core.prompts import (SystemMessagePromptTemplate, 
                                    HumanMessagePromptTemplate,
                                    ChatPromptTemplate)

from langchain_core.output_parsers import StrOutputParser

langfuse_handler = CallbackHandler()
base_url = "http://localhost:11434"
# model = 'qwen3'
model = 'llama3.2'

llm = ChatOllama(base_url=base_url, model=model)

system = SystemMessagePromptTemplate.from_template("""You are helpful AI assistant who answer user question based on the provided context.""")
prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:"""

prompt = HumanMessagePromptTemplate.from_template(prompt)
messages = [system, prompt]
template = ChatPromptTemplate(messages)
qna_chain = template | llm | StrOutputParser()

def ask_llm(context, question):
    return qna_chain.invoke(
        {'context': context, 'question': question},
        config={"callbacks": [langfuse_handler]}
    )
