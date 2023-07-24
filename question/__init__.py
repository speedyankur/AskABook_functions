import logging

import azure.functions as func
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    query = req.params.get('query')
    base = req.params.get('base')
    if not base:
        return func.HttpResponse(
             "Missing knoledge base param.",
             status_code=400
        )
    if not query:
        return func.HttpResponse(
             "Missing query param.",
             status_code=400
        )
    logging.info('base:'+base)
    logging.info('query:'+query)
    
    ACTIVELOOP_TOKEN = os.getenv('ACTIVELOOP_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
    OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
    embeddings = OpenAIEmbeddings(disallowed_special=(), deployment="text-embedding-ada-002", chunk_size = 16)
    logging.info('got embeddings')
    # initialize Deeplake
    db = None
    try:
        db = DeepLake(
            dataset_path="hub://speedyankur/{}".format(base),
            #read_only=True,
            exec_option = "auto",
            embedding_function=embeddings,
        )
    except Exception as err:
        logging.info(f"Unexpected {err=}, {type(err)=}")
    logging.info('got db:')
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20
    logging.info('got retriever')
    BASE_URL = "https://experimenteongpt4.openai.azure.com/"
    OPENAI_GPT4_API_KEY = os.getenv('OPENAI_GPT4_API_KEY')
    DEPLOYMENT_NAME = "gpt-4-32k"
    API_VERSION = "2023-03-15-preview"
    model = AzureChatOpenAI(
        openai_api_base=BASE_URL,
        openai_api_version=API_VERSION,
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=OPENAI_GPT4_API_KEY,
        openai_api_type="azure",
    )
    logging.info('got model:')
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever, return_source_documents=True, verbose=True)
    logging.info('got qa chain')
    result = None
    try:
        result = qa({"question": query, "chat_history": []})
        logging.info('got result')
        urls = []
        for docs in result['source_documents']:
            urls.append(docs.metadata['source'])
        response  = json.dumps({
            'answer': result['answer'],
            'metadata': {
                'url' : urls
            }
        }) 
        return func.HttpResponse(response)
    except Exception as err:
        logging.info(f"chain exception {err=}, {type(err)=}")
        return func.HttpResponse(
             f"chain exception {err=}, {type(err)=}",
             status_code=500
        )
