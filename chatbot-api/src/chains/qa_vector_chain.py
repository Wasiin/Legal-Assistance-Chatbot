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
    index_name="qa_vector",
    node_label="QA",
    text_node_properties=[
        "Question",
        "Answer",
    ],
    embedding_node_property="embedding",
)

qa_template = """หน้าที่ของคุณคือการใช้ข้อมูลข้อกฎหมายที่มีอยู่ในบริบทที่กำหนด
เพื่อให้คำตอบเกี่ยวกับประเด็นหรือสถานการณ์ทางกฎหมาย ใช้บริบทที่ให้มาเพื่อให้คำตอบที่ถูกต้องและครบถ้วนมากที่สุด
ห้ามสร้างข้อมูลเพิ่มเติมที่ไม่ได้ระบุไว้อย่างชัดเจนในบริบท หากคุณไม่มีข้อมูลเพียงพอที่จะตอบคำถาม
โปรดระบุอย่างชัดเจนว่าคุณไม่ทราบ หากต้องการเพิ่มเติมหรือปรับเปลี่ยนในรายละเอียด โปรดแจ้ง!
{context}
ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
"""

qa_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=qa_template)
)

qa_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [qa_system_prompt, qa_human_prompt]

qa_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

qa_vector_chain = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(search_type="similarity", search_kwargs={"k":20}),
    verbose=True,
)
qa_vector_chain.combine_documents_chain.llm_chain.prompt = qa_prompt