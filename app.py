from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from flask import Flask, render_template, request, redirect, url_for
from github import Github
from revChatGPT.V1 import Chatbot
from ignore import should_ignore
import openai
import os
import subprocess
import shutil

app = Flask(__name__)

model_id = "gpt-3.5-turbo"
openai.api_key = "<Key>"
os.environ["OPENAI_API_KEY"] = "<Key>"
chatbot = Chatbot(config={
  "access_token": "<Key>"
})

def langchain_response(question):
    loader = DirectoryLoader(path="./summaries", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Total documents loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    
    db.save_local("faiss_index")
    
    query_results = db.similarity_search(question)
    relevant_documents = [doc.page_content for doc in query_results]

    prompt_template = f"Question: {question}\n\nAnswer:"
    # Create a prompt template
    for idx, doc_content in enumerate(relevant_documents):
        prompt_template += f"Document {idx + 1}: {doc_content}\n\n"
        
    prompt_template += "Based on the above documents, answer the question.\n\n"
    
    response = ""
    for data in chatbot.ask(
        prompt_template,
    ):
        response = data["message"]
    print(response)
    
def get_repo_files_recursive(folder):
    repo_files = []

    for entry in os.scandir(folder):
        if entry.name == ".git":
            continue  # Skip the .git folder

        if entry.is_file():
            file_info = {"path": os.path.join(entry.path), "name": entry.name}
            repo_files.append(file_info)
        elif entry.is_dir():
            subdir = os.path.join(entry.path)
            repo_files.extend(get_repo_files_recursive(subdir))

    return repo_files

def convert_and_move_to_summaries(source_path, target_folder):
    with open(source_path, 'r') as code_file:
        code_content = code_file.read()

    # Get the filename without extension
    file_name = os.path.basename(source_path)
    file_name_without_extension = os.path.splitext(file_name)[0]

    # Create the new text file path
    target_path = os.path.join(target_folder, f"{file_name_without_extension}.txt")

    # Write code content to the new text file
    with open(target_path, "w") as txt_file:
        txt_file.write(code_content)

    print(f"File converted and moved: {target_path}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        github_url = request.form.get('github_url')
        repo_name = github_url.split("/")[-1].replace(".git", "")
        repo_folder = os.path.join('clone', repo_name)

        # Clone the repository
        subprocess.run(["git", "clone", github_url, repo_folder])

        # Process the cloned files
        if not os.path.exists('summaries'):
            os.mkdir('summaries')

        files = get_repo_files_recursive(repo_folder)
        
        for file in files:
            if not should_ignore(file['name']):
                convert_and_move_to_summaries(file['path'], 'summaries')
                with open(file['path'], 'r') as code_file:
                    content = code_file.read()
                summary_prompt = f"Explain the contents of the {file['name']} file in 500 words. Include function names, methods and working of the code: {content}\n\nSummary:"
                response = ""
                for data in chatbot.ask(
                    summary_prompt,
                ):
                    response = data["message"]
                with open(f"summaries/{file['name']}.txt", "w") as f:
                    f.write(response)

                # Print the files and its source after its sent to summarization
                print(f"File: {file['name']} from {file['path']}")

        # Perform Langchain and cleanup
        question = request.form.get('question')
        langchain_response(question)

        # Clean up the cloned repository folder
        shutil.rmtree("./clone")
        shutil.rmtree('summaries')

        return redirect(url_for('index'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)