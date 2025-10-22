import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.node_parser import SentenceSplitter


UPLOAD_FOLDER = 'uploads'
PERSIST_DIR = 'faiss_storage'

print("Mengatur model embedding ke Google GenAI ('gemini-embedding-001')...")

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key is None:
    print("="*50)
    print("ERROR: GOOGLE_API_KEY environment variable not found.")
    print("Harap atur di terminal Anda sebelum menjalankan:")
    print('Contoh: $env:GOOGLE_API_KEY = "AIza..."')
    print("="*50)
    exit() 

Settings.embed_model = GoogleGenAIEmbedding(model_name='gemini-embedding-001', api_key=api_key)
print("Model embedding Google GenAI (gemini-embedding-001) berhasil diatur.")

Settings.llm = GoogleGenAI(model_name="models/gemini-pro", api_key=api_key)
print("Model LLM (GoogleGenAI - gemini-pro) berhasil diatur.")

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

query_engine = None

def load_query_engine():
    global query_engine
    
    docstore_path = os.path.join(PERSIST_DIR, 'docstore.json')
    
    if os.path.exists(docstore_path):
        try:
            print(f"Memuat index dari {PERSIST_DIR}...")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            
            query_engine = index.as_query_engine(similarity_top_k=5)
            
            print("Index berhasil dimuat. Siap menerima kueri.")
            return True
        except Exception as e:
            print(f"Gagal memuat index yang ada: {e}")
            return False
    else:
        print("File index (docstore.json) tidak ditemukan. Harap unggah dokumen untuk membuat index.")
        return False

def index_file(file_path):
    try:
        print(f"Memulai pengindeksan untuk file: {file_path}")
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        
        if not documents:
            print("Tidak ada dokumen yang berhasil dimuat dari file.")
            return

        parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(documents)
        
        print(f"Membuat index dari {len(nodes)} node (potongan)...")
        index = VectorStoreIndex(nodes)
        
        print(f"Menyimpan index ke {PERSIST_DIR}...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("Pengindeksan selesai.")
        
    except Exception as e:
        print(f"Error selama pengindeksan: {e}")
        raise

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    if 'document' not in request.files:
        return jsonify({"error": "Tidak ada file yang terdeteksi"}), 400
    
    file = request.files['document']
    
    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(file_path)
            print(f"File disimpan di: {file_path}")

            index_file(file_path)
            
            load_query_eng_result = load_query_engine()
            
            if not load_query_eng_result:
                print("load_query_engine gagal setelah upload.")
                return jsonify({"error": "File diunggah, tapi gagal memuat query engine."}), 500

            return jsonify({"message": f"File '{filename}' berhasil diunggah dan diindeks!"})

        except Exception as e:
            print(f"Error saat mengunggah: {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def handle_query():
    global query_engine
    
    if query_engine is None:
        return jsonify({"error": "Query engine belum siap. Harap unggah dan indeks dokumen terlebih dahulu."}), 400

    if request.method == 'POST':
        try:
            data = request.json
            user_query = data.get('query')

            if not user_query:
                return jsonify({"error": "Tidak ada kueri yang diberikan"}), 400

            print(f"Memproses kueri: {user_query}")
            response = query_engine.query(user_query)
            final_response = str(response)

            return jsonify({
                "response": final_response
            })

        except Exception as e:
            print(f"Error saat kueri: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    load_query_engine()
    
    app.run(debug=True, port=5000)
