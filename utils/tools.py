import json
import os
import shutil
from glob import glob


def read_json_file(file_path):
    file_path = "./script/"+file_path
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_prompt(query: str, history: list):
    use_message = {"role": "user", "content": query}
    if history is None:
        history = []
    history.append(use_message)
    message = {"text": history}
    return message


def process_response(response_str: str, history: list):
    res_dict: dict = json.loads(response_str)
    code = res_dict.get("header", {}).get("code")
    status = res_dict.get("header", {}).get("status", 2)

    if code == 0:
        res_dict = res_dict.get("payload", {}).get(
            "choices", {}).get("text", [{}])[0]
        res_content = res_dict.get("content", "")

        if len(res_dict) > 0 and len(res_content) > 0:
            # Ignore the unnecessary data
            if "index" in res_dict:
                del res_dict["index"]
            response = res_content

            if status == 0:
                history.append(res_dict)
            else:
                history[-1]["content"] += response
                response = history[-1]["content"]

            return response, history, status
        else:
            return "", history, status
    else:
        print("error code ", code)
        print("you can see this website to know code detail")
        print("https://www.xfyun.cn/doc/spark/%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E.html")
        return "", history, status


def init_script(history: list, jsonfile):
    script_data = read_json_file(jsonfile)
    return script_data


def create_script(name, characters, summary, details):

    import os
    if not os.path.exists("script"):
        os.mkdir("script")
    data = {
        "name": name,
        "characters": characters,
        "summary": summary,
        "details": details
    }
    json_data = json.dumps(data, ensure_ascii=False)
    print(json_data)
    with open(f"./script/{name}.json", "w", encoding='utf-8') as file:
        file.write(json_data)
    pass


def txt2vec(name: str, file_path: str):
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    loader = TextLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=128)
    split_docs = text_splitter.split_documents(data)
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    import sentence_transformers
    EMBEDDING_MODEL = "model/text2vec_ernie/"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings.client = sentence_transformers.SentenceTransformer(
        embeddings.model_name, device='cuda')
    from langchain.vectorstores import FAISS
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(f"data/faiss/{name}/")


def pdf2vec(name: str, file_path: str):
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    split_docs = loader.load_and_split()
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    import sentence_transformers
    EMBEDDING_MODEL = "model/text2vec_ernie/"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings.client = sentence_transformers.SentenceTransformer(
        embeddings.model_name, device='cuda')
    from langchain.vectorstores import FAISS
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(f"data/faiss/{name}/")


# def mycopyfile(srcfile, dstpath):                       # 复制函数
#     if not os.path.isfile(srcfile):
#         print("%s not exist!" % (srcfile))
#     else:
#         fpath, fname = os.path.split(srcfile)
#         print(fpath)
#         print(fname)             # 分离文件名和路径
#         if not os.path.exists(dstpath):
#             os.makedirs(dstpath)                       # 创建路径
#         shutil.copy(srcfile, dstpath + fname)          # 复制文件
#         print("copy %s -> %s" % (srcfile, dstpath + fname))
