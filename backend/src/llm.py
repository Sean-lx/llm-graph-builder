import logging
from langchain.docstore.document import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
import boto3
import google.auth
from src.shared.constants import MODEL_VERSIONS


def get_llm(model_version: str):
    """Retrieve the specified language model based on the model name."""
    env_key = "LLM_MODEL_CONFIG_" + model_version
    env_value = os.environ.get(env_key)
    logging.info("Model: {}".format(env_key))
    if "gemini" in model_version:
        credentials, project_id = google.auth.default()
        model_name = MODEL_VERSIONS[model_version]
        llm = ChatVertexAI(
            model_name=model_name,
            convert_system_message_to_human=True,
            credentials=credentials,
            project=project_id,
            temperature=0,
            safety_settings={
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    elif "openai" in model_version:
        model_name = MODEL_VERSIONS[model_version]
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model_name,
            temperature=0,
        )

    elif "azure" in model_version:
        model_name, api_endpoint, api_key, api_version = env_value.split(",")
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=api_endpoint,
            azure_deployment=model_name,  # takes precedence over model parameter
            api_version=api_version,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

    elif "anthropic" in model_version:
        model_name, api_key = env_value.split(",")
        llm = ChatAnthropic(
            api_key=api_key, model=model_name, temperature=0, timeout=None
        )

    elif "fireworks" in model_version:
        model_name, api_key = env_value.split(",")
        llm = ChatFireworks(api_key=api_key, model=model_name)

    elif "groq" in model_version:
        model_name, base_url, api_key = env_value.split(",")
        llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)

    elif "bedrock" in model_version:
        model_name, aws_access_key, aws_secret_key, region_name = env_value.split(",")
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )

        llm = ChatBedrock(
            client=bedrock_client, model_id=model_name, model_kwargs=dict(temperature=0)
        )

    elif "ollama" in model_version:
        model_name, base_url = env_value.split(",")
        # llm = ChatOpenAI(
        #     api_key="ollama",
        #     model=model_name,
        #     temperature=0,
        #     base_url=base_url + "/v1",
        # )
        llm = ChatOllama(base_url=base_url, model=model_name, temperature=0)

    else:
        model_name = "diffbot"
        llm = DiffbotGraphTransformer(
            diffbot_api_key=os.environ.get("DIFFBOT_API_KEY"),
            extract_types=["entities", "facts"],
        )
    logging.info(f"Model created - Model Version: {model_version}")
    return llm, model_name


def get_combined_chunks(chunkId_chunkDoc_list):
    chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE"))
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list


def get_graph_document_list(
    llm, combined_chunk_document_list, allowedNodes, allowedRelationship
):
    futures = []
    graph_document_list = []
    prompt: ChatPromptTemplate | None = None
    if llm.get_name() == "ChatOllama":
        node_properties = False
        prompt = get_ollama_model_transformer_prompt()
    else:
        node_properties = ["description"]
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        prompt=prompt,
        node_properties=node_properties,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
    )
    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in combined_chunk_document_list:
            chunk_doc = Document(
                page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
            )
            futures.append(
                executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc])
            )

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            graph_document = future.result()
            graph_document_list.append(graph_document[0])

    return graph_document_list


def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
    llm, model_name = get_llm(model)
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
    graph_document_list = get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship
    )
    return graph_document_list

def get_ollama_model_transformer_prompt() -> ChatPromptTemplate :
    system_prompt = """
    You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. 
    Your task is to identify  the entities and relations requested with the user prompt from a given text.  
    """

    human_prompt = """
    Below are a number of examples of text and their extracted entities and relationships.
    # Example 1
    # text: 
    '''
    Adam is a software engineer in Microsoft since 2009, and last year he got an award as the Best Talent.
    '''
    # output:
    [
        {{
            "head": "Adam",
            "head_type": "Person",
            "relation": "WORKS_FOR",
            "tail": "Microsoft",
            "tail_type": "Company",
        }},
        {{
            "head": "Adam",
            "head_type": "Person",
            "relation": "HAS_AWARD",
            "tail": "Best Talent",
            "tail_type": "Award",
        }}
    ]

    # Example 2
    # text:
    '''
    Microsoft is a tech company that provide several products such as Microsoft Word. 
    Microsoft Word is a lightweight app that accessible offline.
    '''
    # output:
    [
        {{
            "head": "Microsoft Word",
            "head_type": "Product",
            "relation": "PRODUCED_BY",
            "tail": "Microsoft",
            "tail_type": "Company",
        }},
        {{
            "head": "Microsoft Word",
            "head_type": "Product",
            "relation": "HAS_CHARACTERISTIC",
            "tail": "lightweight app",
            "tail_type": "Characteristic",
        }},
        {{
            "head": "Microsoft Word",
            "head_type": "Product",
            "relation": "HAS_CHARACTERISTIC",
            "tail": "accessible offline",
            "tail_type": "Characteristic",
        }}
    ]

    # Important Notes:
    1. You must generate the output in a JSON list containing JSON objects. Each object should have the keys: "head", "head_type", "relation", "tail", and "tail_type". 
    2. When extracting entities, it's vital to ensure consistency. If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity. The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
    3. The value strings of the keys "head" and "tail" must be in the same language as the input text.
    4. Do not return any null values for the JSON object.
    5. Do not return any empty strings for the JSON object.
    6. Do not return any entity or relation that is not found in the text.
    7. Do not add any explanations. Just return the pure json list.

    Follow the instructions above strictly.
    Extract entities and relations from the given input text:
    # text:
    '''
    {input}
    '''
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                human_prompt,
            ),
        ]
    )