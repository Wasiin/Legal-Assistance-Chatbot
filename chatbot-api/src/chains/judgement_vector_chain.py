import os
from langchain_neo4j import Neo4jVector
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel




JUDGEMENT_QA_MODEL = os.getenv("JUDGEMENT_QA_MODEL")
#set_llm_cache(InMemoryCache())
#if os.getenv("mode") == "prod"

#llm_model=ChatOllama(model=JUDGEMENT_QA_MODEL, temperature=0,)
llm_model=AzureAIChatCompletionsModel()

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        cache_folder=os.path.join(os.path.dirname(__file__),"embedding_model"),
        model_kwargs={'device': 'mps'}
    ),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="judgment_vector",
    node_label="Judgment",
    text_node_properties=[
        "judgment_id",
        "penalty",
        "operation",
        "issue",
        "date",
        "r_law"
    ],
    embedding_node_property="embedding",
)

judgment_template = """Your job is to use legal information provided 
in the given context to answer questions about legal topics 
or scenarios. Use the following context to provide accurate 
and detailed answers. Be as thorough as possible, 
but do not fabricate any information that is not explicitly 
provided in the context. If you do not have enough 
information to answer a question, state clearly that you do not know.
{context}
ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
"""

judgment_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=judgment_template)
)

judgment_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [judgment_system_prompt, judgment_human_prompt]

judgment_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

judgment_vector_chain = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(search_type="similarity", search_kwargs={"k":20}),
    verbose=True,
)
judgment_vector_chain.combine_documents_chain.llm_chain.prompt = judgment_prompt

