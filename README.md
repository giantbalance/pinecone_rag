# pinecone_rag
RAG using GPT by pinecone

#environment
conda env create -f rag_test.yaml <br/>
conda activate rag

# How to upload PDF 
--upsert "file_name" <br/>
example : <br/>
python3 ./pdfloader_deploy.py --upsert sample_news.pdf

# How to ask GPT
--retrieve "asking questions" <br/>
example : <br/>
python3 ./pdfloader_deploy.py --retrieve "who is bohyung kim and where does he live?"

