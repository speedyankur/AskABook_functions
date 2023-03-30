import logging

import azure.functions as func
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    query = req.params.get('query')
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV # next to api key in console
    )
    #provide the index name, where you hv already uploaded the indexes.. index creation is a separate process.
    index_name = "langchain-openai"
    namespace = "book"

    docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)    
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(query,
    include_metadata=True, namespace=namespace)

    reponse  = chain.run(input_documents=docs, question=query)    
    if query:
        return func.HttpResponse(f"{reponse}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a query in the query string.",
             status_code=200
        )
