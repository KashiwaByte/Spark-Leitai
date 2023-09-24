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
llm = Spark_forlangchain(n=10)
llm_tools = Spark_tools_forlangchain(n=10)

memory1 = ConversationSummaryBufferMemory(llm=llm, max_token_limit=4096)
conversation_1 = ConversationChain(llm=llm, memory=memory1)

memory2 = ConversationSummaryBufferMemory(llm=llm_tools, max_token_limit=4096)
conversation_2 = ConversationChain(llm=llm_tools, memory=memory2)

memory3 = ConversationSummaryBufferMemory(llm=llm_tools, max_token_limit=4096)
template_1 = """
现在是正方立论环节，请你听取正方立论，并针对立论进行反驳。
///正方立论内容如下：{text}///
反驳需要逐条反驳观点和论据，并且要给出详细的理由。
"""
template_2 = """
我们正在进行一场辩论赛，你需要扮演一名反方辩手来和用户完成辩论
现在是你的立论环节，你需要在与用户相反的持方上立论，给出自己的观点,
当用户反驳你时，你需要与他辩驳
///用户回答如下：{text}///
"""
template_3 = """
接下来是对方对你的盘问，请你真诚的回答每一个问题
///用户盘问内容如下：{text}///
"""
template_4 = """
接下来是你对对方的盘问，请你针锋相对地对他提出问题。
///用户回答如下：{text}///
"""
template_5 = """
你是一个资深的逻辑性很强的顶级辩手，请对我的陈述进行反驳，越详细越好，反驳需要逐条反驳观点和论据，并且要给出详细的理由，质疑数据论据要用上常用的方法和句式，从数据合理性，样本代表性，统计方法，数据解读等多个角度进行考虑。质疑学理论据要从权威性，解读方式，是否有对抗学理等多个角度进行考虑。
///如下是我们的话题以及我的观点：{text}///
"""
end_prompt = """
请你对我们的对辩过程进行总结，总结需要包括以下部分：1.对辩主要针对什么进行讨论。2.评价我的对辩能力，需要根据评级原则给出评级，并且给出具体理由。评级原则如下：等级一，缺乏论证的反驳；等级二，自说自话的反驳；等级三，针锋相对的反驳；等级四，正中要害的反驳。3.根据我的对辩能力提出一定的建议。
示例如下：
好的，我来对我们的对辩过程进行总结。
在我们的对辩过程中，我们主要讨论了动物园是否应该被禁止。我认为动物园对动物的福利和权利造成了负面影响，而您则提出了一些质疑，认为动物园中的动物可以享受比野外更安全的生活条件。
我认为您的对辩能力属于等级三，即针锋相对的反驳。您能够对我的观点提出一些质疑和反驳，并且能够给出一些合理的理由。但是，在某些情况下，您可能会使用一些不太恰当的类比来归谬我的观点，这可能会影响到对辩的质量和效果。
鉴于您的对辩能力，我认为您可以进一步提高自己的辩论技巧。您可以通过更多的阅读和学习，提高自己的知识水平和思维能力，从而更好地进行论证和反驳。此外，在使用类比和比喻时，需要更加谨慎，确保它们能够恰当地表达您的观点，而不会歪曲或归谬对方的观点。
"""

common_prompt = ChatPromptTemplate.from_template
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
    global QA_chain
    QA_chain = RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(
        search_kwargs={"k": 2}), memory=memory3)
    file_paths = [file.name for file in files]
    return file_paths


def Debatebytext_(prompt, help):
    msg = prompt_template.format_prompt(text=prompt).to_string()
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
    global prompt_template
    prompt_template = prompt_1
    progress(0, desc="Starting...")
    time.sleep(1)
    for i in progress.tqdm(range(240), desc="用户立论倒计时中"):
        time.sleep(1)


def ai_init_function(text, progress=gr.Progress()):
    global prompt_template
    prompt_template = prompt_1
    progress(0, desc="Starting...")
    time.sleep(1)
    for i in progress.tqdm(range(240), desc="AI立论倒计时中"):
        time.sleep(1)


def ask_function(text, progress=gr.Progress()):
    global prompt_template
    prompt_template = prompt_2
    progress(0, desc="Starting...")
    time.sleep(1)
    for i in progress.tqdm(range(90), desc="用户质询倒计时中"):
        time.sleep(1)


def ai_ask_function(text, progress=gr.Progress()):
    global prompt_template
    prompt_template = prompt_3
    progress(0, desc="Starting...")
    time.sleep(1)
    for i in progress.tqdm(range(150), desc="AI质询倒计时中"):
        time.sleep(1)


def again_function(text, progress=gr.Progress()):
    global prompt_template
    prompt_template = prompt_4
    progress(0, desc="Starting...")
    time.sleep(1)
    for i in progress.tqdm(range(150), desc="对辩倒计时中"):
        time.sleep(1)


def free_function(text, progress=gr.Progress()):
    global prompt_template
    prompt_template = prompt_5
    progress(0, desc="Starting...")
    time.sleep(1)
    for i in progress.tqdm(range(240), desc="自由辩倒计时中"):
        time.sleep(1)


def end_function(text, help):
    msg = end_prompt
    response = conversation_1.run(msg)
    help.append((text, response))
    return help, help


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
        init_btn = gr.Button("用户立论")
        ai_init_btn = gr.Button("AI立论")
        ask_btn = gr.Button("用户质询")
        ai_ask_btn = gr.Button("AI质询")
        again_btn = gr.Button("对辩")
        free_btn = gr.Button("自由辩")
        end_btn = gr.Button("结辩")
        init_btn.click(init_function, text, text2)
        ask_btn.click(ask_function, text, text2)
        ai_ask_btn.click(ai_ask_function, text, text2)
        again_btn.click(again_function, text, text2)
        free_btn.click(free_function, text, text2)
        end_btn.click(end_function, [text, state], [chatbot, state])

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
        init_btn = gr.Button("用户立论")
        ai_init_btn = gr.Button("AI立论")
        ask_btn = gr.Button("用户质询")
        ai_ask_btn = gr.Button("AI质询")
        again_btn = gr.Button("对辩")
        free_btn = gr.Button("自由辩")
        end_btn = gr.Button("结辩")
        init_btn.click(init_function, text, text2)
        ask_btn.click(ask_function, text, text2)
        ai_ask_btn.click(ai_ask_function, text, text2)
        again_btn.click(again_function, text, text2)
        free_btn.click(free_function, text, text2)
        end_btn.click(end_function, [text, state], [chatbot, state])

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
        init_btn = gr.Button("用户立论")
        ai_init_btn = gr.Button("AI立论")
        ask_btn = gr.Button("用户质询")
        ai_ask_btn = gr.Button("AI质询")
        again_btn = gr.Button("对辩")
        free_btn = gr.Button("自由辩")
        end_btn = gr.Button("结辩")
        init_btn.click(init_function, text, text2)
        ask_btn.click(ask_function, text, text2)
        ai_ask_btn.click(ai_ask_function, text, text2)
        again_btn.click(again_function, text, text2)
        free_btn.click(free_function, text, text2)
        end_btn.click(end_function, [text, state], [chatbot, state])
    with gr.Row():
        file_output = gr.File(label='请上传你想要, 目前支持txt、docx、md格式',
                              file_types=['.txt', '.md', '.docx'])
        upload_button = gr.UploadButton("Click to Upload a File", scale=1, file_types=[
            "text"])
        upload_button.upload(upload_file, upload_button, file_output)


demo = gr.TabbedInterface([Debate_page, Debate_classic_page, Debate_text_page], [
                          "辩论赛事",  "经典辩论赛事", "自定义材料赛事"])
demo.queue(concurrency_count=3).launch()
