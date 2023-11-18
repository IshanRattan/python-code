from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import gradio as gr
import torch
import time
import re

class Prompts():

    _SYSTEM_PROMPT = """You are a teacher teaching your students in an interactive way, by providing correct feedback to their responses so as to guide them step by step to the correct answer. Your feedback should not use any part of the solution to the problem. Never forget our task! The problem is:
STUDENT:
TEACHER:
SYSTEM: Do you think TEACHER has really given the correct answer to the STUDENT?

Never forget you are a teacher and I am a STUDENT. Never flip roles!
You must help me to correct my errors and misunderstandings to complete the task. Never provide me with the final solution directly!
Do not agree with the STUDENT when the STUDENT gives a wrong answer.

Your response should always start with:
Teacher: YOUR_RESPONSE

YOUR_RESPONSE should be imitating human teachers. You can ask questions like 'So what should you do next?', 'Can you calculate . . . ?', 'Are you sure you need to add here?', 'Why do you think you need to add these numbers?', 'Can you go walk me through your solution?'

Your job is to find out my errors in my solution and correct me. You must use questions or comments to guide me to help my thoughts.
Always end YOUR_RESPONSE with: Please continue to attempt.
NEVER give me responses that are not related to the problem I asked and do give me new
Your ONLY goal is to help me get a correct solution. Once I reach the correct solution, you MUST end our chat by replying a single word PROBLEM_SOLVED. Please make your response precise and short."""


    B_INST, E_INST = "[INST]", "{} [/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n\n"
    B_SEN, E_SEN = "<s>", "</s>"

    def __init__(self):
        self._language_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                                    device='cuda:0')

    def _check_sort_order(self, query: str) -> str:

        def _fetch_result(res: list) -> str:
            return sorted(res, key=lambda x: x[2])[0]

        pairs = [("largest to smallest", True), ("smallest to largest", False),
                ("highest to lowest", True), ("lowest to highest", False),
                ("biggest to smallest", True), ("smallest to biggest", False),
                ("ascending order", True), ("descending order", False),
                ("increasing order", True), ("decreasing order", False)]
        results = []
        for pair in pairs:
            embeddings = self._language_model.encode([query, pair[0]])
            score = cosine_similarity([embeddings[0]], embeddings[1:])
            distance = levenshtein_distance(query, pair[0])
            results.append((pair, score[0][0], distance))

        results = sorted(results, key=lambda x: x[1])
        return _fetch_result(results[-2:])[0][1]

    def _intent_checker(self, query: str) -> (str, bool):

        numbers = re.findall(r'\d+\d+', query)
        target_words = query.lower().split()
        # other_words = ['number sort']
        order_keywords = ['order', 'ordered', 'ordering', 'arrange', 'arranged', 'arranging', 'sort', 'sorted', 'sorting']
        # number_keywords = ['number', 'numbers', 'integer', 'integers', 'digit', 'digits', 'values']
        direction_keywords = ['small', 'smaller', 'smallest', 'lowest', 'highest', 'large', 'larger', 'largest',
                              'biggest', 'ascending', 'descending', 'increasing', 'decreasing']

        if any(word in target_words for word in order_keywords) and len(numbers) > 1:
            for number in numbers: filtered_query = query.replace(number, '')

            # if any(word in target_words for word in number_keywords): number_kw = True
            # else: number_kw = False

            if any(word in target_words for word in direction_keywords):
                sorted_numbers = sorted(list(map(int, numbers)), reverse=self._check_sort_order(filtered_query))
                return f'The result is {" ".join(list(map(str, sorted_numbers)))}', True
            else:
                return query, False
        else:
            return query, False

    def _prompt_builder(self, history: list[list]) -> str:
      if isinstance(history, str):
          prompt = self.B_SEN + self.B_INST + " " + self.B_SYS + self._SYSTEM_PROMPT + self.E_SYS + self.E_INST.format(history)
          return prompt
      else:
          prompt = self.B_SEN + self.B_INST + " " + self.B_SYS + self._SYSTEM_PROMPT + self.E_SYS + self.E_INST.format(history[0][0])
          for idx in range(0, len(history)):
            if history[idx][1]:
              prompt += """ {} </s>{}""".format(" ".join(history[idx][1].split()), '\n')
              if idx < len(history) - 2:
                prompt += """<s>[INST] {} [/INST]""".format(history[idx + 1][0])
              else:
                prompt += """<s>[INST] {} [/INST]""".format(f'"Teacher: {history[idx][1].replace("Teacher: ", "")} \
                Student: {history[idx + 1][0]}" \
                Has the Student given correct response to the Teacher?')

              # prompt += """ {} </s>\
              #               <s>[INST] {} [/INST]""".format(" ".join(history[idx][1].split()),
              #                                              history[idx + 1][0])
          print(prompt)
          return prompt

    def _fetch_prompt(self, user_input: str or list[list]) -> str:
          return self._prompt_builder(user_input)


class Llama(Prompts):
    def __init__(self, model_name: str):
        super().__init__()

        _config = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_compute_dtype=torch.float32)
        self._model = AutoModelForCausalLM.from_pretrained(model_name,
                                                           torch_dtype=torch.bfloat16)
        # quantization_config=_config)
        self._model.eval()
        self._model.to("cuda:0")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(self, text: str) -> torch.Tensor:  # Check type
        return self._tokenizer(text, return_tensors='pt')

    def _generate_probs(self, input,
                        temperature,
                        max_new_tokens,
                        top_p,
                        top_k):
        return self._model.generate(input['input_ids'].to("cuda:0"), 
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k)

    def _generate_text(self, probs):
        return self._tokenizer.batch_decode(probs, skip_special_tokens=True)[0]

    def run(self, query: str or list[list],
            temperature=1,
            max_new_tokens=300,
            top_p=.95,
            top_k=50) -> str:
        prompt = query  # self._fetch_prompt(query)
        tokenized = self._tokenize(prompt)
        probs = self._generate_probs(tokenized,
                                     temperature=temperature,
                                     max_new_tokens=max_new_tokens,
                                     top_p=top_p,
                                     top_k=top_k)
        answer = self._generate_text(probs)
        return answer.split('[/INST]')[-1].strip()


prompts = Prompts()


llama = Llama("meta-llama/Llama-2-13b-chat-hf")

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = llama.run(history[-1][0])
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(debug=True)

