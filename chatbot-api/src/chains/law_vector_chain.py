import os

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_ollama import ChatOllama

from langchain.chains import RetrievalQA
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

import dotenv
dotenv.load_dotenv()

if os.getenv("MODE") == "cloud":
    llm_model = AzureAIChatCompletionsModel()
else: 
    llm_model = ChatOllama(model=os.getenv("OLLAMA_MODEL"),temperature=0)

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        cache_folder=os.path.join(os.path.dirname(__file__),"embedding_model"),
        model_kwargs={'device': 'mps'}
    ),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="law_vector",
    node_label="Raw",
    text_node_properties=[
        "raw_name",
        "description",
    ],
    embedding_node_property="embedding",
)

law_template = """หน้าที่ของคุณคือการใช้ข้อมูลข้อกฎหมายที่มีอยู่ในบริบทที่กำหนด
เพื่อให้คำตอบเกี่ยวกับประเด็นหรือสถานการณ์ทางกฎหมาย ใช้บริบทที่ให้มาเพื่อให้คำตอบที่ถูกต้องและครบถ้วนมากที่สุด
ห้ามสร้างข้อมูลเพิ่มเติมที่ไม่ได้ระบุไว้อย่างชัดเจนในบริบท หากคุณไม่มีข้อมูลเพียงพอที่จะตอบคำถาม
โปรดระบุอย่างชัดเจนว่าคุณไม่ทราบ หากต้องการเพิ่มเติมหรือปรับเปลี่ยนในรายละเอียด โปรดแจ้ง!
{context}
ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
"""

law_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=law_template)
)

law_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [law_system_prompt, law_human_prompt]

law_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

law_vector_chain = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(search_type="similarity", search_kwargs={"k":20}),
    verbose=True,
)
law_vector_chain.combine_documents_chain.llm_chain.prompt = law_prompt