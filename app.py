import os
import torch
from werkzeug.utils import secure_filename
from langchain_groq import ChatGroq
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.documents import Document

app = Flask(__name__)

## Configuration
load_dotenv()
UPLOAD_FOLDER = 'pdfs'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CHROMA_DIR = "./chroma_db"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'device': device}
    )

def get_vectorstore(embeddings):
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        return Chroma.from_documents(documents=[], embedding=embeddings, persist_directory=CHROMA_DIR)

def process_and_add_pdfs(filenames, embeddings, vectorstore):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_docs = []
    for filename in filenames:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            split_docs = splitter.split_documents(docs)
            for doc in split_docs:
                doc.metadata['source'] = filename
            new_docs.extend(split_docs)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    if new_docs:
        vectorstore.add_documents(new_docs)
        # vectorstore.persist()  # Not needed with langchain_chroma
    return len(new_docs)

def initialize_rag(vectorstore):
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")

    system_prompt = """
You are a helpful assistant. Use the provided context from PDF documents to answer the user's question as accurately and helpfully as possible.
- If the answer is directly available, provide it.
- If the answer requires combining, summarizing, or inferring from multiple parts of the context, do so.
- If the answer is not explicitly in the context, look for similar references or related information in the context and use them to construct the most relevant and helpful answer.
- If you can reasonably infer an answer, provide it and mention that it is inferred.
- If the answer truly cannot be found or inferred from the context, say: "I could not find the answer in the provided documents."
- Be clear, concise, and conversational.
- Reference the document or section if possible.
- Make your answer as concise as possible, using at most 5 sentences.
Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}, search_type="similarity")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    contexualize_q_system_prompt = """
    Reformulate the user's question to be standalone based on chat history,
    but ONLY if the referenced information exists in the PDF documents.
    Never reference information outside the PDF context.
    If the question references something not in the documents, ask for clarification.
    """


    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", contexualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=chat_prompt
    )

    final_rag_chain = create_retrieval_chain(
        history_aware_retriever,
        create_stuff_documents_chain(llm=llm, prompt=prompt)
    )

    return final_rag_chain

# Global variables
embeddings = get_embeddings()
vectorstore = get_vectorstore(embeddings)
rag_chain = initialize_rag(vectorstore)
chat_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files[]')
    uploaded_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                file.save(filepath)
                uploaded_files.append(filename)
            else:
                # If file already exists, skip re-upload
                uploaded_files.append(filename)

    if not uploaded_files:
        return jsonify({'error': 'No valid PDF files uploaded'}), 400

    # Only process and add new PDFs
    processed_count = process_and_add_pdfs(uploaded_files, embeddings, vectorstore)
    global rag_chain
    rag_chain = initialize_rag(vectorstore)

    return jsonify({
        'message': f'Successfully processed {processed_count} document(s)',
        'files': uploaded_files
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    global chat_history, rag_chain

    if rag_chain is None:
        return jsonify({'error': 'Please upload PDF documents first'}), 400

    data = request.get_json()
    user_input = data.get('question')

    if not user_input:
        return jsonify({'error': 'No question provided'}), 400

    try:
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response["answer"])
        ])

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        return jsonify({
            'question': user_input,
            'answer': response["answer"]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    global chat_history
    chat_history = []
    return jsonify({'status': 'success'})

@app.route('/list-documents', methods=['GET'])
def list_documents():
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
        return jsonify({'documents': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
