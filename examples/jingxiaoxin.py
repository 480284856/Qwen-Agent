import os
import pprint
import random
import string
import logging
import datetime
import speech_recognition as sr

from typing import *
from collections import deque
from qwen_agent.agent import Agent
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message

from qwen_agent.gui.web_ui import *
from speech_recognition import AudioData
from qwen_agent.multi_agent_hub import MultiAgentHub
from AIDCM.frontend.audio_gradio.zijie_tts import AudioProducer,AudioConsumer
from AIDCM.frontend.audio_gradio.ali_stt import Recognition,Callback,lingji_stt_gradio,lingji_stt_gradio_va

class myWebUI:
    """A Common chatbot application for agent."""

    def __init__(self, agent: Union[Agent, MultiAgentHub, List[Agent]], chatbot_config: Optional[dict] = None):
        """
        Initialization the chatbot.

        Args:
            agent: The agent or a list of agents,
                supports various types of agents such as Assistant, GroupChat, Router, etc.
            chatbot_config: The chatbot configuration.
                Set the configuration as {'user.name': '', 'user.avatar': '', 'agent.avatar': '', 'input.placeholder': '', 'prompt.suggestions': []}.
        """
        chatbot_config = chatbot_config or {}

        if isinstance(agent, MultiAgentHub):
            self.agent_list = [agent for agent in agent.nonuser_agents]
            self.agent_hub = agent
        elif isinstance(agent, list):
            self.agent_list = agent
            self.agent_hub = None
        else:
            self.agent_list = [agent]
            self.agent_hub = None

        user_name = chatbot_config.get('user.name', 'user')  # 用户的名称，会展示到Web界面上。
        self.user_config = {
            'name': user_name,
            'avatar': chatbot_config.get(                    # 用户头像路径。
                'user.avatar',
                get_avatar_image(user_name),
            ),
        }

        self.agent_config_list = [{
            'name': agent.name,
            'avatar': chatbot_config.get(
                'agent.avatar',
                get_avatar_image(agent.name),
            ),
            'description': agent.description or "I'm a helpful assistant.",             # 获取agent的描述信息，这个可以在agent初始化时设置。
        } for agent in self.agent_list]

        self.input_placeholder = chatbot_config.get('input.placeholder', '跟我聊聊吧～')  # 放在输入框里的文字，但不会作为输入。
        self.prompt_suggestions = chatbot_config.get('prompt.suggestions', [])          # 输入建议
        self.verbose = chatbot_config.get('verbose', False)

        self.text_queue:deque = gr.State(deque([]))   # 异步TTS：文本队列
        self.old_total_response:str = ''              # 异步TTS：从开始到上一个时间点的总响应
        self.audio_queue:deque = gr.State(deque([]))  # 异步TTS：音频队列

        self.producer_audio_thread_active = False     # 异步TTS：音频生产者线程是否激活
        self.consumer_audio_thread_active = False     # 异步TTS：音频消费者线程是否激活
    """
    Run the chatbot.

    Args:
        messages: The chat history.
    """

    def run(self,
            messages: List[Message] = None,
            share: bool = False,
            server_name: str = None,
            server_port: int = None,
            concurrency_limit: int = 10,
            enable_mention: bool = False,
            **kwargs):
        self.run_kwargs = kwargs


        # 定制Gradio界面主题
        customTheme = gr.themes.Default(
            primary_hue=gr.themes.utils.colors.blue,    # 主题颜色。
            radius_size='sm',                           # 图标按钮的圆角大小。
        )

        # 使用自定义主题和CSS样式启动Gradio界面
        with gr.Blocks(            
                css=os.path.join(os.path.dirname(__file__), 'assets/appBot.css'),   # CSS: 专注于Web显示效果优化
                theme=customTheme,
        ) as demo:
            history = gr.State([])                # 一直生效的历史记录
            # 布局对话界面
            with gr.Row(elem_classes='container'):
                # 对话栏
                with gr.Column(scale=4):          
                    chatbot = mgr.Chatbot(
                        value=convert_history_to_chatbot(messages=messages),
                        avatar_images=[
                            self.user_config,
                            self.agent_config_list,
                        ],
                        height=None,              # 自适应高度。
                        avatar_image_width=40,
                        flushing=True,            # 更加流畅的流式对话，会一个个字地显示。
                        flushing_speed=5,         # 流式输出速度。1～10.
                        show_copy_button=False,   # 复制对话框中的内容，显示了不太好看
                    )
                    # 在聊天框下方，添加一个输入框和一个按钮，点击按钮，可以进行语音输入。
                    with gr.Row(elem_classes='container'):
                        with gr.Column(scale=9):
                            input = mgr.MultimodalInput(placeholder=self.input_placeholder,)
                        with gr.Column(scale=1):
                            submit = gr.Button(value="microphone")
                            submit.click(
                                fn=lingji_stt_gradio,
                                inputs=[input],
                                outputs=[input]
                            ).then(
                                fn=self.add_text,
                                inputs=[input, chatbot, history],
                                outputs=[input, chatbot, history],
                                queue=False
                            ).then(
                                self.agent_run,
                                [chatbot, history],
                                [chatbot, history],
                            ).then(   
                                self.flushed,
                                None, 
                                [input]  
                            )

                            tab = gr.Button("语音唤醒模式")
                            tab.click(
                                self.lingji_stt_gradio_voice_awake,
                                inputs=[input, chatbot, history],
                                outputs=[input, chatbot, history]

                            )
                            
                # 侧边栏，设置右侧选择代理和插件的区域
                with gr.Column(scale=1):          
                    if len(self.agent_list) > 1:  # 当存在多个代理时，添加代理选择下拉列表
                        agent_selector = gr.Dropdown(
                            [(agent.name, i) for i, agent in enumerate(self.agent_list)],
                            label='Agents',
                            info='选择一个Agent',
                            value=0,
                            interactive=True,
                        )

                    # 创建Agent信息块
                    agent_info_block = self._create_agent_info_block()        
                    # 创建代理插件展示块
                    agent_plugins_block = self._create_agent_plugins_block()  

                    if self.prompt_suggestions:
                        gr.Examples(
                            label='推荐对话',
                            examples=self.prompt_suggestions,
                            inputs=[input],
                        )

                # 当agent选择发生变动时，更新agent的配置
                if len(self.agent_list) > 1:
                    agent_selector.change(
                        fn=self.change_agent,
                        inputs=[agent_selector],
                        outputs=[agent_selector, agent_info_block, agent_plugins_block],
                        queue=False,
                    )

                # 当用户点击提交时，执行add_text函数。
                input_promise = input.submit(            # 当用户点击提交,或者是在输入后按下回车时,触发这个函数.
                    fn=self.add_text,
                    inputs=[input, chatbot, history],
                    outputs=[input, chatbot, history],   # 把函数的返回结果赋值给哪些变量.
                    queue=False,                         # If True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
                )

                if len(self.agent_list) > 1 and enable_mention:
                    input_promise = input_promise.then(
                        self.add_mention,
                        [chatbot, agent_selector],
                        [chatbot, agent_selector],
                    ).then(
                        self.agent_run,
                        [chatbot, history, agent_selector],
                        [chatbot, history, agent_selector],
                    )
                else:
                    # 在submit函数运行完成后,运行agent_run函数
                    # 相当于: input.submit().then(agent_run)
                    input_promise = input_promise.then(
                        self.agent_run,
                        [chatbot, history],
                        [chatbot, history],
                    )

                # 在agent_run函数运行完成后,运行flushed函数
                # 该函数会解锁输入框,把其锁定状态设置为False
                input_promise.then(self.flushed, None, [input])

            # 这个方法允许你在 Blocks 组件加载时运行一个函数，并将其输出用于初始化界面上的组件。
            # 但传入None, 应该是没有什么作用的,所以可以注释掉.
            demo.load(None)

        # 配置并发限制并启动Gradio界面
        # 启用排队机制,可以防止系统过载，并确保每个请求都能被按顺序处理。
        # .queue()方法会返回self,也就是对象自己.
        demo.queue(default_concurrency_limit=concurrency_limit).launch(share=share,
                                                                       server_name=server_name,
                                                                       server_port=server_port)

    def change_agent(self, agent_selector):
        yield agent_selector, self._create_agent_info_block(agent_selector), self._create_agent_plugins_block(
            agent_selector)

    def add_text(self, _input, _chatbot, _history):
        """
        向对话历史中添加用户输入文本及相关信息。
        
        参数:
        - _input:mgr.MultimodalInput 用户的输入对象，包含文本和可能的文件信息。
        - _chatbot:mgr.Chatbot[list[list[None, None]]] 对话机器人对象，用于存储和处理对话相关数据。
        - _history: gr.State[list] 对话历史列表，用于记录用户的输入和机器人的回复。
        
        返回:
        - 一个生成器，用于在对话过程中更新界面和保持对话状态。
        """

        # step1: 添加用户的输入
        _history.append({
            ROLE: USER,
            CONTENT: [{
                'text': _input.text
            }],
        })

        # 如果用户配置了名称，则将这个名称添加到
        if self.user_config[NAME]:
            _history[-1][NAME] = self.user_config[NAME]

        # 处理用户输入中的文件，将图片和其它文件分别添加到新增的历史消息项中
        if _input.files:
            for file in _input.files:
                if file.mime_type.startswith('image/'):
                    _history[-1][CONTENT].append({IMAGE: 'file://' + file.path})
                else:
                    _history[-1][CONTENT].append({FILE: file.path})

        _chatbot.append([_input, None])
            
        # gr.update: 更新输入框的值的value属性为None, interactive为False
        # 设置interactive为False是为了在模型输出的过程中不允许用户继续输入
        yield gr.update(interactive=False, value=None), _chatbot, _history

    def add_mention(self, _chatbot, _agent_selector):
        if len(self.agent_list) == 1:
            yield _chatbot, _agent_selector

        query = _chatbot[-1][0].text
        match = re.search(r'@\w+\b', query)
        if match:
            _agent_selector = self._get_agent_index_by_name(match.group()[1:])

        agent_name = self.agent_list[_agent_selector].name

        if ('@' + agent_name) not in query and self.agent_hub is None:
            _chatbot[-1][0].text = '@' + agent_name + ' ' + query

        yield _chatbot, _agent_selector

    def agent_run(self, _chatbot, _history, _agent_selector=None):
        """
        控制代理运行的函数。

        - param _chatbot:mgr.Chatbot[list[list[None, None]]] 会话历史记录的列表，每个元素是一个二元组.
        - param _history:gr.State[list] 用于代理决策的历史消息列表。
        - param _agent_selector: 用于选择下一个代理的函数，如果为None，则默认选择第一个代理。
        - yield: 在每个代理运行后生成聊天机器人状态、历史和当前选择的代理索引。
        """
        
        # 先启动音频生产者和消费者线程，这样就可以达到监听的效果。
        self.producer_audio_thread(self.text_queue, self.audio_queue)
        self.consumer_audio_thread(self.audio_queue)

        # 第一次按下语音按钮的时候,input是None,所以如果把None放进去的话,会导致Web界面上出现空的用户问题,所以在这里进行拦截
        # 如果最新一轮 用户的输入是空的,则说明是语音输入时 第一次点击录音按钮 进入这里的,此时我们选择跳过,并删除最新一轮的对话.
        if _chatbot[-1][0] and _chatbot[-1][0].text:
            # 如果设置为详细模式，则输出函数输入信息
            if self.verbose:
                logger.info('agent_run input:\n' + pprint.pformat(_history, indent=2))

            last_response_text = None
            _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]

            agent_runner = self.agent_list[_agent_selector or 0]   # 返回第_agent_selector个agent或者是第0个
            if self.agent_hub:
                agent_runner = self.agent_hub
            response = []
            for response in agent_runner.run(_history, **self.run_kwargs):  # 大模型推理结果: [{'role': 'assistant', 'content': '你好'}]
                if not response:
                    continue
                display_response = convert_fncall_to_text(response)         # [{'role': 'assistant', 'content': '你好', 'name': None}]

                if display_response is None or len(display_response) == 0:
                    continue

                agent_name, response_text = (
                    display_response[-1][NAME],
                    display_response[-1][CONTENT],
                )
                if response_text is None:
                    continue
                elif response_text == PENDING_USER_INPUT:                  # 表示接下来应该是用户的输入.
                    logger.info('Interrupted. Waiting for user input!')
                    continue

                # TODO: Remove this `are_similar_enough`. This hack is not smart.
                if last_response_text is not None and not are_similar_enough(last_response_text, response_text):  # 如果响应文本和上一次响应文本不相似，则添加一个空消息到聊天机器人中。
                    _chatbot.append([None, None])
                    _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]

                # 更新聊天机器人的代理响应列表
                agent_index = self._get_agent_index_by_name(agent_name)
                _chatbot[-1][1][agent_index] = response_text
                last_response_text = response_text

                if len(self.agent_list) > 1:
                    _agent_selector = agent_index

                # 根据_agent_selector是否存在，决定是否生成带索引的yield输出
                # 在yield出去的时候,chatbot对象会被更新,所以yield执行之后,Web界面才会更新.
                if _agent_selector is not None:
                    yield _chatbot, _history, _agent_selector
                else:
                    stream_response:str = _chatbot[-1][-1][0]  # 大模型的当前时刻的推理结果
                    self.producer_text(stream_response)
                    yield _chatbot, _history
            if response:
                _history.extend([res for res in response if res[CONTENT] != PENDING_USER_INPUT])

            if _agent_selector is not None:
                yield _chatbot, _history, _agent_selector
            else:                
                yield _chatbot, _history

            if self.verbose:
                logger.info('agent_run response:\n' + pprint.pformat(response, indent=2))
            self.producer_text(None)  # 推理完成后，在队列中加入None表示推理完成

            # 当第二次点击microphone按钮时，表示录音停止，接下来需要做的就是等待音频生产者和消费者线程结束。
            self._producer_audio_thread.join()
            self._consumer_audio_thread.join()
            self.old_total_response = ''
        else:
            del _chatbot[-1]
            del _history[-1]
            yield _chatbot, _history

    def flushed(self):
        from qwen_agent.gui.gradio import gr

        return gr.update(interactive=True)

    def _get_agent_index_by_name(self, agent_name):
        if agent_name is None:
            return 0

        try:
            agent_name = agent_name.strip()
            for i, agent in enumerate(self.agent_list):
                if agent.name == agent_name:
                    return i
            return 0
        except Exception:
            print_traceback()
            return 0

    def _create_agent_info_block(self, agent_index=0):
        from qwen_agent.gui.gradio import gr

        agent_config_interactive = self.agent_config_list[agent_index]

        return gr.HTML(
            format_cover_html(
                bot_name=agent_config_interactive['name'],
                bot_description=agent_config_interactive['description'],
                bot_avatar=agent_config_interactive['avatar'],
            ))

    def _create_agent_plugins_block(self, agent_index=0):
        from qwen_agent.gui.gradio import gr

        agent_interactive = self.agent_list[agent_index]

        if agent_interactive.function_map:
            capabilities = [key for key in agent_interactive.function_map.keys()]
            return gr.CheckboxGroup(
                label='插件',
                value=capabilities,
                choices=capabilities,
                interactive=False,
            )

        else:
            return gr.CheckboxGroup(
                label='插件',
                value=[],
                choices=[],
                interactive=False,
            )
    
    # 异步TTS代码
    def producer_text(self, response:str=None):
        '''把模型的response放入到text队列中'''
        if response != None:
            # 每次拿到reponse时，使用新的response去掉old_total_response，得到response_delta，这个就是在当前时间点得到的response。
            response_delta = self.remove_first_match(response, self.old_total_response)

            # 我们对response_delta进行这样的操作，从左到右，判断其是否有有‘，。！？’等符号，如果有的话，就从左到右，从开头到这个符号为止，截取这段文本，然后把这段文本放入到text队列中，以及拼接到old_total_response对象右侧。如果没有，则不做操作。
            while response_delta:
                # 找到第一个标点符号的位置
                ## i for i in xxx, 叫做生成器表达式，使用圆括号包裹生成器表达式，表示这是一个generator，而使用[]或{}的话，则表示是一个列表或集合表达式，
                ## 前者不会把for表达式执行完，然后返回结果，而是返回一个generator对象，只有在便利这个对象（比如使用next方法）的时候才会执行for表达式，然后返回结果。
                ## 而列表或集合表达式会立即执行for表达式，然后返回结果。
                ## 返回generator的方式称为惰性计算，只有在需要的时候才把值放到内存中。每次迭代时，生成器都会返回一个值，然后记住当前位置，下次遍历这个生成器时，则会从之前记住的位置的下一个位置开始便利。
                ## 不把英文的'.'算入截断标点符号中，因为大模型生成的文本中，标题会用到'.'，比如‘6.’
                punctuation_index = next((i for i, char in enumerate(response_delta) if char in [',' , '，', '。', '！', '？', '!', '?']), -1)
                if punctuation_index != -1:                          # 如果生成器表达式不是空的，即找的到第一个标点符号
                    text = response_delta[:punctuation_index + 1]    # 截取这段文本
                    self.old_total_response += text                  # 拼接到old_total_response对象右侧
                    self.text_queue.value.append(text)               # 把这段文本放入到text队列中
                    
                    # 如果response_delta中存在不止一个符号，那么我们在做完第一个符号对应的工作后，把response_delta更新为去掉第一段文本的剩下文本，然后进行同样的操作。
                    response_delta = response_delta[punctuation_index + 1:]
                
                # 如果response_delta没有‘，。！？’等符号了，则不做操作。
                if punctuation_index==-1:
                    break
        else:  # 生成完成后，会往队列中放入一个None，表示生成完毕。
            self.text_queue.value.append(response)
            print(self.text_queue)

    def producer_audio_thread(self, _text_queue:deque, _audio_queue:deque):
        '''
        使用AIDCM中的AudioProducer类，创建一个生产者线程，用于将文本转换为音频。
        '''
        print("producer_audio_thread is called")
        # 维护一个self.producer_audio_thread_active变量，用于判断生产者线程是否已经启动。
        ## 如果self.producer_audio_thread_active为True，则表示生产者线程已经启动，否则表示生产者线程尚未启动。
        ## self.producer_audio_thread_active默认为False，在第一次点击microphone按钮时，我们可以把其设置为True，表示开始生产者线程。
        if not self.producer_audio_thread_active:
            self.producer_audio_thread_active = True
            self._producer_audio_thread = AudioProducer(_text_queue, _audio_queue)
            self._producer_audio_thread.start()
        else:
            # 当第二次点击microphone按钮时，表示录音停止，接下来需要做的就是等待音频生产者线程结束。
            self.producer_audio_thread_active = False
               
    def consumer_audio_thread(self, _audio_queue: deque):
        '''
        使用AIDCM中的AudioConsumer类，创建一个消费者线程，用于播放音频。
        '''
        # 维护一个self.consumer_audio_thread_active变量，用于判断消费者线程是否已经启动。
        ## 如果self.consumer_audio_thread_active为True，则表示消费者线程已经启动，否则表示消费者线程尚未启动。
        ## self.consumer_audio_thread_active默认为False，在第一次点击microphone按钮时，我们可以把其设置为True，表示开始消费者线程。
        if not self.consumer_audio_thread_active:
            self.consumer_audio_thread_active = True
            self._consumer_audio_thread = AudioConsumer(_audio_queue)
            self._consumer_audio_thread.start()
        else:
            # 当第二次点击microphone按钮时，表示录音停止，接下来需要做的就是等待音频消费者线程结束。
            self.consumer_audio_thread_active = False

    def remove_first_match(self, s:str, sub_s:str):
        '''从s中删除第一次出现的sub_s'''
        if sub_s in s:
            return s.replace(sub_s, '', 1)
        else:
            return s

    # 语音唤醒代码
    def get_logger(self, ):
        # 日志收集器
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        # 设置控制台处理器，当logger被调用时，控制台处理器额外输出被调用的位置。
        # 创建一个控制台处理器并设置级别为DEBUG
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # 创建一个格式化器，并设置格式包括文件名和行号
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
        ch.setFormatter(formatter)

        # 将处理器添加到logger
        logger.addHandler(ch)

        return logger

    def get_random_file_name(self, length=20, extension='.wav'):
        '''create a random file name with current time'''
        current_time = datetime.datetime.now().strftime("%Y-%m-%d%H-%M-%S")
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        return f"{current_time}_{random_string}{extension}"

    def save_audio_file(self, audio:AudioData, sample_rate=16000):
        file_name = self.get_random_file_name()
        with open(file_name, "wb") as f:
            f.write(audio.get_wav_data(convert_rate=sample_rate))
        return file_name

    def recognize_speech(self, audio_path, recognizer:Recognition) ->str:
        return recognizer.call(audio_path).get_sentence()[0]['text']

    def lingji_stt_gradio_voice_awake(self, 
                                      input_box:mgr.MultimodalInput,
                                      chatbot:mgr.Chatbot,
                                      history:list):
        '''
        语音唤醒模块
        ---

        input_box: modelscope_studio的MultimodalInput
        chatbot:mgr.Chatbot[list[list[None, None]]] 对话机器人对象，用于存储和处理对话相关数据。
        history: gr.State[list] 对话历史列表，用于记录用户的输入和机器人的回复。
        '''
        recognizer = sr.Recognizer()
        logger = self.get_logger()

        recognition= Recognition(
                    model="paraformer-realtime-v1",            # 语音识别模型
                    format='wav',                     # 音频格式
                    sample_rate=16000,                # 指定音频的采样率，16000表示每秒采样16000次。
                    callback=Callback()
                    )
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, 1)          # 调整背景噪音
            
            while True:
                try:
                    # 一直监听语音
                    logger.info("Listening for wake word 'hei siri'...")
                    # audio = recognizer.listen(source)               # 监听麦克风
                    logger.info("Recognizing done.")
                    # audio_path = self.save_audio_file(audio)
                    # result = self.recognize_speech(audio_path, recognizer=recognition)
                    # os.remove(audio_path)
                    # logger.info(f"Recognized: {result}")
                    result = "123"

                    # 当用户说出特定唤醒词时
                    if "" in result:
                        logger.info("Wake word detected!")

                        # TODO: 给出固定的欢迎回复
                        input_box.text = "你好！"
                        _, chatbot, history = next(iter(self.add_text(
                            _input=input_box,
                            _chatbot=chatbot,
                            _history=history
                        )))

                        # 实时语音识别
                        input_box.text = lingji_stt_gradio_va()
                        chatbot, history = next(iter(self.agent_run(
                            _chatbot=chatbot,
                            _history=history
                        )))

                        input_box = self.flushed()
                        
                        break
                except sr.UnknownValueError:
                    continue
                except TypeError as e:
                    continue
            
            return input_box, chatbot, history
def llm_config():
    # 步骤 1：配置您所使用的 LLM。
    llm_cfg = {
        # 使用 DashScope 提供的模型服务：
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': 'sk-8deaaacf2fb34929a076dfc993273195',
        # 如果这里没有设置 'api_key'，它将读取 `DASHSCOPE_API_KEY` 环境变量。

        # 使用与 OpenAI API 兼容的模型服务，例如 vLLM 或 Ollama：
        # 'model': 'Qwen2-7B-Chat',
        # 'model_server': 'http://localhost:8000/v1',  # base_url，也称为 api_base
        # 'api_key': 'EMPTY',

        # （可选） LLM 的超参数：
        'generate_cfg': {
            'top_p': 0.8
        }
    }
    return llm_cfg

def create_agent(llm_cfg):
    # 步骤 2：创建一个智能体。这里我们以 `Assistant` 智能体为例.
    system_instruction = '''你是一个乐于助人的AI助手。你总是用中文回复用户。'''
    bot = Assistant(llm=llm_cfg,
                    system_message=system_instruction,
                    )
    return bot

def chat_cli(bot):
    # 步骤 3：作为聊天机器人运行智能体。
    messages = []  # 这里储存聊天历史。
    while True:
        # 例如，输入请求 "你好"。
        query = input('用户请求: ')
        # 将用户请求添加到聊天历史。
        messages.append({'role': 'user', 'content': query})
        response = []
        for response in bot.run(messages=messages):
            # 流式输出。
            print('机器人回应:')
            pprint.pprint(response, indent=2)
        # 将机器人的回应添加到聊天历史。
        messages.extend(response)

def chat_webui(
        agent,
        chatbot_config = None
):
    myWebUI(agent, chatbot_config=chatbot_config).run(messages=[{'role': 'assistant', 'content': [{'text': '你好，旅行者。'}]}])

if __name__ == "__main__":
    chatbot_config = {
        'prompt.suggestions': [    # 放在Web界面右边的一些建议输入，点击这些输入，会自动放在文本输入框内，
            '你今天真好看！',        # 但是还需要点击发送按钮才可以发送。
            '晚上去吃好吃的嘛~',
            '宝贝，你又瘦啦！！',
        ]
    }

    llm_cfg = llm_config()
    bot = create_agent(llm_cfg)
    chat_webui(
        agent=bot,
        chatbot_config=chatbot_config
    )
