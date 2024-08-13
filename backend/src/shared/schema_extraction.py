from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from src.llm import get_llm
from src.shared.constants import MODEL_VERSIONS
from langchain_core.prompts import ChatPromptTemplate

class Schema(BaseModel):
    """Knowledge Graph Schema."""

    labels: List[str] = Field(description="list of node labels or types in a graph schema")
    relationshipTypes: List[str] = Field(description="list of relationship types in a graph schema")

PROMPT_TEMPLATE_WITH_SCHEMA = (
    "You are an expert in schema extraction, especially for extracting graph schema information from various formats."
    "Generate the generalized graph schema based on input text. Identify key entities and their relationships and "
    "provide a generalized label for the overall context"
    "Schema representations formats can contain extra symbols, quotes, or comments. Ignore all that extra markup."
    "Only return the string types for nodes and relationships. Don't return attributes."
)

PROMPT_TEMPLATE_WITHOUT_SCHEMA = (
"""You are a top-tier algorithm designed for extracting information from unstructured texts into structured formats to build a knowledge graph. 
Your task is to identify the entities and relations requested with the user prompt from a given text.
    
Below are a number of examples of text and their extracted entities and relationships.
# Example 1
# text: 
'''
Adam is a software engineer in Microsoft since 2009, and last year he got an award as the Best Talent.
'''
# output:
{{
    "labels": ["Person", "Company", "Award"],
    "relationship_types": ["WORKS_FOR", "HAS_AWARD"]
}}

# Example 2
# text:
'''
Microsoft is a tech company that provide several products such as Microsoft Word. 
Microsoft Word is a lightweight app that accessible offline.
'''
# output:
{{
    "labels": ["Company", "Product", "Characteristic"],
    "relationship_types": ["PRODUCED_BY", "HAS_CHARACTERISTIC"]
}}
    
# Important Notes:
1. Ensure you use available types for node labels. Ensure you use basic or elementary types for node labels. For example, when you identify an entity representing a person, always label it as 'Person'. Avoid using more specific terms like 'mathematician' or 'scientist'.
2. Relationships represent connections between entities or concepts. Ensure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary types such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. Make sure to use general and timeless relationship types!
3. Do not add any explanations. Just return the JSON object.
4. Do not return duplicate labels.
"""
)

def schema_extraction_from_text(input_text:str, model:str, is_schema_description_cheked:bool):
    
    llm, model_name = get_llm(model)
    if is_schema_description_cheked:
        schema_prompt = PROMPT_TEMPLATE_WITH_SCHEMA
    else:
        schema_prompt = PROMPT_TEMPLATE_WITHOUT_SCHEMA
        
    prompt = ChatPromptTemplate.from_messages(
    [("system", schema_prompt), ("user", "{text}")]
    )
    
    runnable = prompt | llm.with_structured_output(
        schema=Schema,
        method="function_calling",
        include_raw=False,
    )
    
    raw_schema = runnable.invoke({"text": input_text})
    return raw_schema