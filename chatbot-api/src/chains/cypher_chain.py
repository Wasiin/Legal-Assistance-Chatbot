import os
#from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel


JUDGEMENT_QA_MODEL = os.getenv("JUDGEMENT_QA_MODEL")
JUDGEMENT_CYPHER_MODEL = os.getenv("JUDGEMENT_CYPHER_MODEL")
llm_model=AzureAIChatCompletionsModel()
#set_llm_cache(InMemoryCache())

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

graph.refresh_schema()

cypher_generation_template = """

Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

# Request information on the Judgment "1938/2563"
MATCH (n:Judgment)
WHERE n.judgment_id = "1938/2563"
RETURN n

# Request for law information "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 573"
MATCH (n:Raw)
WHERE n.raw_name = "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 573"
RETURN n.description

# What laws does the Judgment "1938/2020" relate to?
MATCH (n:Judgment)
WHERE n.judgment_id = "1938/2563"
RETURN n.r_raw

# Judgment information related to the law "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 573"
MATCH (n:Judgment)
WHERE "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 573" IN n.r_raw
RETURN n

Make sure to use IS NULL or IS NOT NULL when analyzing missing properties.
Make sure use structure CASE...WHEN...THEN...END
Never return embedding properties in your queries.
You must never include the statement "GROUP BY" in your query.


The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a user's natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, assume the query
results are in units of days unless otherwise specified.

When listing names in search results, such as legal information names,
be wary of names that contain commas or other punctuation marks.
For example [ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 386", "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 573",
"ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 686", "ประมวลกฎหมายวิธีพิจารณาความแพ่ง มาตรา 142 (5)"]
are names of multiple laws and sections. Make sure you list all the names in a way
that is not ambiguous and allows others to provide the full name.

Never say you don't have the right information if there is data in
the query results. Always use the data in the query results.

Helpful Answer:
ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

judgement_cypher_chain = GraphCypherQAChain.from_llm(
    #cypher_llm=ChatOllama(model=JUDGEMENT_CYPHER_MODEL, temperature=0,),
    #qa_llm=ChatOllama(model=JUDGEMENT_QA_MODEL, temperature=0,),
    cypher_llm=llm_model,
    qa_llm=llm_model,
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=20,
    allow_dangerous_requests=True,
)