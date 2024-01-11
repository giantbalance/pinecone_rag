from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from numpy import vectorize
import openai
import getpass
import pinecone
import argparse

PINECONE_API_KEY = getpass.getpass("Pinecone API Key: ")
PINECONE_API_ENV = getpass.getpass("Pinecone API ENV: ")
OPENAI_API_KEY = getpass.getpass("API KEY: " )

MODEL = "text-embedding-ada-002"
# Default Index Name
INDEX_NAME = "mypine"


#### Function Setting #####
# Define a function to create embeddings
def create_embeddings(texts):
    embeddings_list = []
    for text in texts:
        res = openai.embeddings.create(input=[text], model=MODEL)
        embeddings_list.append(res.data[0].embedding)
    return embeddings_list

# Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings, ids, metadatas):
    index.upsert(vectors=[(id, embedding, metadata ) for id,embedding, metadata in zip(ids, embeddings,metadatas)])

def pdf_Loader(pdf_path):
    # create a loader
    loader = PyPDFLoader(pdf_path)

    # load your data
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    return documents


#### Environment Setting #####
#Pine Cone Setting
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to API key in console
)
#Open API Setting
openai.api_key = OPENAI_API_KEY


def upsertIndex(file_name):
    # create a pinecone index
    index_name = INDEX_NAME
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    pinecone_index = pinecone.Index(index_name)

    # TODO : Args to upsert vector or not
    # PDF Loading to pinecone
    pdf_name = file_name

    documents = pdf_Loader(pdf_name)
    texts = [str(doc) for doc in documents] 

    # create embeddings
    embeddings = create_embeddings(texts)
    # TODO: extract text field only. current metadata stores all data in the documents
    metadatas = [{'text' : str(doc)} for doc in documents]
    # Upsert
    ids = list(map(str,range(len(texts))))
    pinecone_index.upsert(vectors=[(id, embedding, metadata ) for id,embedding, metadata in zip(ids, embeddings,metadatas)])

def retrieve(query_text):
    #retrieve in pinecone.
    # sample query : who is bohyung kim and where does he live?
    res = openai.embeddings.create(input=[query_text], model=MODEL)
    ctx = res.data[0].embedding
    index_name = INDEX_NAME
    pinecone_index = pinecone.Index(index_name)
    query_ctx = pinecone_index.query(ctx, top_k=1, include_metadata=True)

    #export text
    contexts = [
        x['metadata']['text'] for x in query_ctx['matches']
    ]

    #build prompt
    prompt_start = (
        "Answer the question based on the context below. \n\n" +
        "Context : \n"
    )
    prompt_end = (
        f"\n\n Question : {query_text} \n Answer: "
    )

    #1000 token = 3,750 words
    limit = 3750
    for i in range(0, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )

    print(f"This is the prompt {prompt}")

    # ask open ai 
    stream = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role" :"user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


def main():
    print(f"PDF Load and Retrival using GPT")
    parser = argparse.ArgumentParser(description='Load PDF File and Retrieve')
    parser.add_argument('--upsert', metavar='PDF_FILE', type=str, help='Perform upsert operation on a PDF file')
    parser.add_argument('--retrieve', metavar='QUERY', type=str, help='Perform retrieve query')

    args = parser.parse_args()

    if args.upsert:
        upsertIndex(args.upsert)
    elif args.retrieve:
        retrieve(args.retrieve)
    else:
        print('Please specify either --upsert or --retrieve option.')

if __name__ == "__main__":
    main()



