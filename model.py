# from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
# import os
# import openai
# from flask import Flask
# from openai import OpenAI

# client=OpenAI(api_key = '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')
# pdf_file_path=r"C:\Users\GB Tech\Desktop\mahabharatdt.pdf"
# embedding_persistant_directory="C:/Users/GB Tech/Downloads/Embedding_vector"
# loader = PyPDFLoader(pdf_file_path)
# pages = loader.load() 
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# chunks = text_splitter.split_documents(pages) 
# chunks[0].page_content 
# embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

# db = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=embedding_persistant_directory)

# query='Mahabharat' 
# chunks_with_query_context = db.similarity_search(query,k=1)

# print("Search Results:")
# for i, result in enumerate(chunks_with_query_context, 1):
#     print(f"Result {i}:\n {result}\n") 
# def search_chromadb(db, question, k):
#     try:
#         if not db:
#             raise ValueError("Database object is not provided.")
#         if not question or not isinstance(question, str):
#             raise ValueError("Invalid question input.")
#         if not isinstance(k, int) or k < 1:
#             raise ValueError("Invalid value for 'k'. 'k' should be an integer greater than or equal to 1.")
#         docs = db.similarity_search(question, k=k)
#         if not docs:
#             print(f"No documents found in the database for the query: {question}")
#             return []
#         results = [doc.page_content for doc in docs if doc.page_content]
#         if not results:
#             print("No non-empty results from semantic search.")
#         print(f"\nChunks from semantic search: {results}")
#         return results
#     except Exception as e:
#         print(f"An error occurred: {str(e)}") 
#         return [] 
# def generate_answer(question, context):
#     context = context
#     debug = False
#     if debug:
#         print("Context:\n" + context)
#         print("\n\n")
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"Please ask another question related to Query and Data\"\n\n"},
#                 {"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
#             ],
#             temperature=0,
#             max_tokens=500,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0,
#             stop=None,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return "" 
    


# loaded_db=Chroma(persist_directory=embedding_persistant_directory,embedding_function=embedding)
# print("Question Answeting System    ")
# while True:
#         user_query = input("Enter your question (type 'quit' or 'Q' or 'q' to exit): ").strip().lower()
#         if user_query == 'quit' or user_query == 'q' or user_query == 'Q':
#             print("Visit Again ")
#             break
#         contexts = search_chromadb(loaded_db, user_query,1)  
#         answer = generate_answer(user_query, contexts) 
#         print("\n Matching Answer:", answer) 
#         print("\n wait and enter next question")

from flask import Flask, request, jsonify,render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import openai
import os
import pdfplumber



openai.api_key = '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'
embedding_persistent_directory = "C:/Users/GB Tech/Downloads/Embedding_vector"

app = Flask(__name__)
db = None  # Initialize the global variable

def load_and_create_chroma(pdf_content):
    try:
        with pdfplumber.open(pdf_content) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        chroma_db = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=embedding_persistent_directory)
        return chroma_db
    except Exception as e:
        print(f"An error occurred while loading and creating chroma: {str(e)}")
        return None

def search_chromadb(db, question, k):
    try:
        if not db:
            raise ValueError("Database object is not provided.")
        if not question or not isinstance(question, str):
            raise ValueError("Invalid question input.")
        if not isinstance(k, int) or k < 1:
            raise ValueError("Invalid value for 'k'. 'k' should be an integer greater than or equal to 1.")
        docs = db.similarity_search(question, k=k)
        if not docs:
            print(f"No documents found in the database for the query: {question}")
            return []
        results = [doc.page_content for doc in docs if doc.page_content]
        if not results:
            print("No non-empty results from semantic search.")
        print(f"\nChunks from semantic search: {results}")
        return results
    except Exception as e:
        print(f"An error occurred: {str(e)}") 
        return [] 

def generate_answer(question, context):
    context = context
    debug = False
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"Please ask another question related to Query and Data\"\n\n"},
                {"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
            ],
            temperature=0,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""

@app.route('/upload', methods=['POST'])
def upload_file():
    global db
    try:
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No file provided"}), 400 
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        query = request.form.get('query', '')
        if file:
            pdf_content = file.read()
            db = load_and_create_chroma(pdf_content)
            if db is not None:
                chunks_with_query_context = db.similarity_search(query, k=1)
                print("Search Results:")
                for i, result in enumerate(chunks_with_query_context, 1):
                    print(f"Result {i}:\n {result}\n")

                return jsonify({"message": "File processed successfully"})
            else:
                return jsonify({"error": "Failed to process file"}), 500
        else:
            return jsonify({"error": "Failed to process file"}), 500
          
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        user_query = request.json['question'].strip().lower()
        if user_query == 'quit' or user_query == 'q':
            return jsonify({"message": "Visit Again"})
        contexts = search_chromadb(db, user_query, 1)
        answer = generate_answer(user_query, contexts)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Question Answering System")
    app.run(debug=True)

































# from flask import Flask, request,jsonify
# from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
# import os
# import openai
# from flask import Flask
# from openai import OpenAI


# openai.api_key = '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' 
# embedding_persistent_directory = "C:/Users/GB Tech/Downloads/Embedding_vector"
 

# app = Flask(__name__)  
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     global db  # Access the global db variable
#     try:
#         if 'pdf_file' not in request.files:
#             return jsonify({"error": "No file provided"}), 400

#         file = request.files['pdf_file']
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         if file:
#             pdf_file_path = os.path.join("C:/path/to/uploaded_pdfs", file.filename)
#             file.save(pdf_file_path)

#             db = load_and_create_chroma(pdf_file_path)

#             # Perform a sample query on the newly created Chroma database
#             query = 'Mahabharat'
#             chunks_with_query_context = db.similarity_search(query, k=1)

#             print("Search Results:")
#             for i, result in enumerate(chunks_with_query_context, 1):
#                 print(f"Result {i}:\n {result}\n")

#             return jsonify({"message": "File uploaded successfully"})
#         else:
#             return jsonify({"error": "Failed to upload file"}), 500
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    

# def load_and_create_chroma(pdf_file_path):
#     loader = PyPDFLoader(pdf_file_path)
#     pages = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = text_splitter.split_documents(pages)
#     embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
#     chroma_db = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=embedding_persistent_directory)
#     return chroma_db

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     try:
#         user_query = request.json['question'].strip().lower()
#         if user_query == 'quit' or user_query == 'q':
#             return jsonify({"message": "Visit Again"})
#         contexts = search_chromadb(db, user_query, 1)
#         answer = generate_answer(user_query, contexts)
#         return jsonify({"answer": answer})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# def get_answer():
#     user_query = request.form.get('question')
#     return {"answer": generate_answer(user_query, "sample_context")}

# def search_chromadb(db, question, k):
#     try:
#         if not db:
#             raise ValueError("Database object is not provided.")
#         if not question or not isinstance(question, str):
#             raise ValueError("Invalid question input.")
#         if not isinstance(k, int) or k < 1:
#             raise ValueError("Invalid value for 'k'. 'k' should be an integer greater than or equal to 1.")
#         docs = db.similarity_search(question, k=k)
#         if not docs:
#             print(f"No documents found in the database for the query: {question}")
#             return []
#         results = [doc.page_content for doc in docs if doc.page_content]
#         if not results:
#             print("No non-empty results from semantic search.")
#         print(f"\nChunks from semantic search: {results}")
#         return results
#     except Exception as e:
#         print(f"An error occurred: {str(e)}") 
#         return [] 
# def generate_answer(question, context):
#     context = context
#     debug = False
#     if debug:
#         print("Context:\n" + context)
#         print("\n\n")
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"Please ask another question related to Query and Data\"\n\n"},
#                 {"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
#             ],
#             temperature=0,
#             max_tokens=500,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0,
#             stop=None,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return "" 
    
# loaded_db=Chroma(persist_directory_embedding_persistant_directory,embedding_function=embedding)
# print("Question Answeting System    ")
# while True:
#         user_query = input("Enter your question (type 'quit' or 'Q' or 'q' to exit): ").strip().lower()
#         if user_query == 'quit' or user_query == 'q' or user_query == 'Q':
#             print("Visit Again ")
#             break
#         contexts = search_chromadb(loaded_db, user_query,1)  
#         answer = generate_answer(user_query, contexts) 
#         print("\n Matching Answer:", answer) 
#         print("\n wait and enter next question")

# if __name__ == '__main__':
#     print("Question Answering System")
#     app.run(debug=True)


