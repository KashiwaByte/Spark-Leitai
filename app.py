from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from utils.API import Spark_forlangchain
from utils.API import Spark_tools_forlangchain
import gradio as gr
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import sentence_transformers
import time
global llm
global conversation_1
global prompt_template

llm = Spark_forlangchain(n=10)
llm_tools = Spark_tools_forlangchain(n=10)

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=4096)
conversation_1 = ConversationChain(llm=llm)

memory = ConversationSummaryBufferMemory(llm=llm_tools, max_token_limit=4096)
conversation_2 = ConversationChain(llm=llm_tools)

template_1 = "现在是正方立论环节，请你听取正方立论，并针对立论进行质询。正方立论内容如下：{text}"
template_2 = ""
template_3 = ""
template_4 = ""
template_5 = ""
template_6 = ""
prompt_1 = ChatPromptTemplate.from_template(template_1)
prompt_2 = ChatPromptTemplate.from_template(template_2)
prompt_3 = ChatPromptTemplate.from_template(template_3)
prompt_4 = ChatPromptTemplate.from_template(template_4)
prompt_5 = ChatPromptTemplate.from_template(template_5)


def init_knowledge_vector_store(filepath):
    EMBEDDING_MODEL = "model/text2vec_ernie/"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings.client = sentence_transformers.SentenceTransformer(
        embeddings.model_name, device='cuda')
    loader = TextLoader(filepath)
    docs = loader.load()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def upload_file(files):
    vector_store = init_knowledge_vector_store(files.name)
    memory_text = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=4096)
    global QA_chain
    QA_chain = RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(
        search_kwargs={"k": 2}), memory=memory_text)
    file_paths = [file.name for file in files]
    return file_paths


def Debatebytext_(prompt, help):
    msg = prompt
    response = QA_chain.run(msg)
    help.append((prompt, response))
    return help, help


def Debate_(prompt, help):
    msg = prompt_template.format_prompt(text=prompt).to_string()
    response = conversation_1.run(msg)
    help.append((prompt, response))
    return help, help


def Debate_classic(prompt, help):
    msg = prompt_template.format_prompt(text=prompt).to_string()
    response = conversation_2.run(msg)
    help.append((prompt, response))
    return help, help


def init_function(text, progress=gr.Progress()):
    progress(0, desc="Starting...")
    time.sleep(1)
    prompt_template = prompt_1
    for i in progress.tqdm(range(240), desc="立论倒计时中"):
        time.sleep(1)


def ask1_function(text, progress=gr.Progress()):
    progress(0, desc="Starting...")
    time.sleep(1)
    prompt_template = prompt_2
    for i in progress.tqdm(range(90), desc="盘问倒计时中"):
        time.sleep(1)


def ask2_function(text, progress=gr.Progress()):
    progress(0, desc="Starting...")
    time.sleep(1)
    prompt_template = prompt_3
    for i in progress.tqdm(range(150), desc="质询倒计时中"):
        time.sleep(1)


def again_function(text, progress=gr.Progress()):
    progress(0, desc="Starting...")
    time.sleep(1)
    prompt_template = prompt_4
    for i in progress.tqdm(range(150), desc="对辩倒计时中"):
        time.sleep(1)


def free_function(text, progress=gr.Progress()):
    progress(0, desc="Starting...")
    time.sleep(1)
    prompt_template = prompt_5
    for i in progress.tqdm(range(240), desc="自由辩倒计时中"):
        time.sleep(1)

title1 = "<h1 style='font-size: 40px;'><center>欢迎体验辩论赛事！</center></h1>"
content1 = "<h1 style='font-size: 20px;'><center>您可以体验限时的辩论全流程</center></h1>"
title2 = "<h1 style='font-size: 40px;'><center>欢迎体验经典辩论赛事！</center></h1>"
content2 = "<h1 style='font-size: 20px;'><center>您可以体验在经典赛题上与可能你了解的大佬们对辩</center></h1>"
title3 = "<h1 style='font-size: 40px;'><center>欢迎体验自定义材料赛事！</center></h1>"
content3 = "<h1 style='font-size: 20px;'><center>您可以自行上传某位大佬的辩论记录来体验与他们对辩</center></h1>"
with gr.Blocks(css="#chatbot{height:300px} .overflow-y-auto{height:500px}") as Debate_page:
    gr.Markdown(title1)
    gr.Markdown(content1)
    state = gr.State([])
    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")
        text2 = gr.Textbox()
    with gr.Row():
        text = gr.Textbox()
        send = gr.Button("🚀 发送")
        send.click(Debate_, [text, state], [chatbot, state])
    with gr.Row():
        init_btn = gr.Button("立论")
        ask1_btn = gr.Button("盘问")
        ask2_btn = gr.Button("质询")
        again_btn = gr.Button("对辩")
        free_btn = gr.Button("自由辩")
        end_btn = gr.Button("结辩")
        init_btn.click(init_function, text, text2)
        ask1_btn.click(ask1_function, text, text2)
        ask2_btn.click(ask2_function, text, text2)
        again_btn.click(again_function, text, text2)
        free_btn.click(free_function, text, text2)

with gr.Blocks(css="#chatbot{height:300px} .overflow-y-auto{height:500px}") as Debate_classic_page:
    gr.Markdown(title2)
    gr.Markdown(content2)
    state = gr.State([])
    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")
        text2 = gr.Textbox()
    with gr.Row():
        text = gr.Textbox()
        send = gr.Button("🚀 发送")
        send.click(Debate_classic, [text, state], [chatbot, state])
    with gr.Row():
        init_btn = gr.Button("立论")
        ask1_btn = gr.Button("盘问")
        ask2_btn = gr.Button("质询")
        again_btn = gr.Button("对辩")
        free_btn = gr.Button("自由辩")
        end_btn = gr.Button("结辩")
        init_btn.click(init_function, text, text2)
        ask1_btn.click(ask1_function, text, text2)
        ask2_btn.click(ask2_function, text, text2)
        again_btn.click(again_function, text, text2)
        free_btn.click(free_function, text, text2)

with gr.Blocks(css="#chatbot{height:300px} .overflow-y-auto{height:500px}") as Debate_text_page:
    gr.Markdown(title3)
    gr.Markdown(content3)
    state = gr.State([])
    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")
        text2 = gr.Textbox()
    with gr.Row():
        text = gr.Textbox()
        send = gr.Button("🚀 发送")
        send.click(Debatebytext_, [text, state], [chatbot, state])
    with gr.Row():
        init_btn = gr.Button("立论")
        ask1_btn = gr.Button("盘问")
        ask2_btn = gr.Button("质询")
        again_btn = gr.Button("对辩")
        free_btn = gr.Button("自由辩")
        end_btn = gr.Button("结辩")
        init_btn.click(init_function, text, text2)
        ask1_btn.click(ask1_function, text, text2)
        ask2_btn.click(ask2_function, text, text2)
        again_btn.click(again_function, text, text2)
        free_btn.click(free_function, text, text2)
    with gr.Row():
        file_output = gr.File(label='请上传你想要, 目前支持txt、docx、md格式',
                          file_types=['.txt', '.md', '.docx'])
        upload_button = gr.UploadButton("Click to Upload a File", scale=1, file_types=[
                                    "text"])
        upload_button.upload(upload_file, upload_button, file_output)



demo = gr.TabbedInterface([Debate_page, Debate_classic_page, Debate_text_page], [
                          "辩论赛事",  "经典辩论赛事","自定义材料赛事"])
demo.queue().launch()
