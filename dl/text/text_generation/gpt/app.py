from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from utils.helper import action, transcribe
from prompts.sysPrompts import sys_prompt
from prompts import promptTemplates
from typing import Iterator
from configs import config
from utils.ui import JS
import gradio as gr
import time
import os


os.environ['OPENAI_API_KEY'] = config.openaiApiKey

class Bot():

    def __init__(self):

        self.llm = ChatOpenAI(temperature=config.temperature,
                         model=config.llmVersion,
                         max_tokens=config.maxTokens)

    def predict(self, message: str, history: list) -> str:
        history_langchain_format = []
        history_langchain_format.append(SystemMessage(content=sys_prompt))
        for human, ai in history:
            if human and ai:
                ai = ai.split('Translated:')[0]
                history_langchain_format.append(HumanMessage(content=human))
                history_langchain_format.append(AIMessage(content=ai))

        if len(history) < 2:
            input = promptTemplates.firstPrompt.format(question=message)
        else:
            if len(history) > 2:
                context = """STUDENT: {}
                STUDENT: {}
                TEACHER: {}""".format(history[0][0], history[-2][0], history[-2][1])
                input = promptTemplates.prompt.format(orig_problem=context, question=message)
            else:
                input = promptTemplates.firstPrompt.format(question=history[-1][0])

        # print(input)
        history_langchain_format.append(HumanMessage(content=input))
        gpt_response = self.llm(history_langchain_format)
        return gpt_response.content


with gr.Blocks(theme=gr.themes.Base(), title=config.tabTitle) as demo:
    tutor = Bot()
    gr.Markdown(JS.appTitle())
    gr.Markdown(config.markdownTxt)

    chatBot = gr.Chatbot(height=config.chatBotHeight)

    msg = gr.Textbox(placeholder=config.msgBoxPlaceholder,
                     container=config.msgBoxContainerVisible,
                     scale=config.msgBoxScale)

    audioBox = gr.Audio(label=config.audioBoxLabel,
                        sources=config.audioBoxSource,
                        type=config.audioBoxSaveType,
                        visible=config.audioBoxVisible,
                        elem_id=config.audioBoxId)

    with gr.Row():
        audioBtn = gr.Button(config.audioBtnLabel,
                             elem_id=config.audioBtnId)
        clear = gr.Button(config.btnClearText)


    def user(userMessage: str, history: list) -> (str, list):
        return "", history + [[userMessage, None]]


    def bot(history: list) -> Iterator[str]:

        botMessage = tutor.predict(history[-1][0], history)
        history[-1][1] = ""
        # translated_text, wav, sr = translator.predict(bot_message, "t2tt", 'spa', src_lang='eng')
        # out = """{}

        # Translated: {}""".format(bot_message, translated_text)

        for character in botMessage:
            history[-1][1] += character
            time.sleep(0.01)
            yield history


    audioBtn.click(fn=action, inputs=audioBtn, outputs=audioBtn).\
        then(fn=lambda x: None,
             inputs=audioBox,
             js=JS.recordAudio()).\
        then(fn=lambda :None,
             js=JS.validateState()).\
        success(fn=transcribe,
             inputs=audioBox,
             outputs=msg).\
        then(fn=lambda: None,
             inputs=None,
             outputs=audioBox,
             queue=False).\
        then(user,
             [msg, chatBot],
             [msg, chatBot],
             queue=False).\
        then(bot,
             chatBot,
             chatBot)

    msg.submit(user, [msg, chatBot], [msg, chatBot], queue=False).then(
        bot, chatBot, chatBot
    )
    clear.click(lambda: None, None, chatBot, queue=False)

demo.queue().launch(share=False)
