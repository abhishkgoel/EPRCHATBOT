from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper,ServiceContext,load_index_from_storage
from langchain import OpenAI
import warnings
import sys
import os
import gradio as gr
import openai
import streamlit as st

apikey= st.secrets["apikey"]
openai.api_key = apikey
os.environ["OPENAI_API_KEY"]=apikey
max_input = 4096
tokens = 256
chunk_size = 600
max_chunk_overlap = .20

prompt_helper = PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size)

#define LLM
llmPredictor = LLMPredictor(llm=OpenAI(temperature =0, model_name='text-ada-001',max_tokens=tokens))

#load data
docs = SimpleDirectoryReader("/home/user/app/New_folder").load_data()

#create vector index
service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=prompt_helper)
vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)

warnings.filterwarnings("ignore")
prompt1 = "Enter your text here"

c=0
storage_context=None
vIndex=None
query_engine=None
def openai_create(prompt):
    global c
    global storage_context
    global vIndex
    global query_engine
    response=None
    if(c==0):
        c=1
        response="The following is a conversation with an EPR AI assistant.\nSelect one category from the following you want to proceed with -\n\nA) Plastic waste \nB) E-WASTE "
    elif(c==1):
        if(prompt.lower()=='a' or prompt.lower()=='plastic waste'):
            c=2
            storage_context = vectorIndex.storage_context.from_defaults(persist_dir='vectorIndex.json')
            vIndex = load_index_from_storage(storage_context)
            query_engine = vIndex.as_query_engine()
            response = "You have successfully selected Plastic Waste.\nEnter you query."
        elif(prompt.lower()=='b' or prompt.lower()=='e-waste'):
            c=2
            storage_context = vectorIndex.storage_context.from_defaults(persist_dir='e-vectorIndex.json')
            vIndex = load_index_from_storage(storage_context)
            query_engine = vIndex.as_query_engine()
            response = "You have successfully selected E-Waste.\nEnter you query."
        else:
            response="Please enter valid category.\nSelect one category from the following you want to proceed with -\n\nA) Plastic waste \nB) E-WASTE "
    elif(c==2):
        if(prompt.lower()!='bye'):
            response = query_engine.query(prompt)
            response = query_engine.query(prompt)
            response = (f"{response}\n")
        elif(prompt.lower()=='bye'):
            c=0
            response = "Bye,See You Soon.."

    return response

def chatgpt_clone(input, history):
    global s
    global inp
    global output
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = openai_create(s[-1])
    history.append((input, output))
    return history, history

block = gr.Blocks()


with block:
    gr.Markdown("""<h1><center>Virtual AI CHAT-EPR</center></h1>""")
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder=prompt1)
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state], queue=False)
    submit.click(lambda x: gr.update(value=''), [ ],[message])
    message.submit(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])
    message.submit(lambda x: gr.update(value=''), [ ],[message])

block.launch()
