import os
import pprint
import re
from typing import List, Optional, Union

from qwen_agent import Agent, MultiAgentHub
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.gui.gradio_utils import format_cover_html
from qwen_agent.gui.utils import (are_similar_enough, convert_fncall_to_text, convert_history_to_chatbot,
                                  get_avatar_image)
from qwen_agent.llm.schema import CONTENT, FILE, IMAGE, NAME, ROLE, USER, Message
from qwen_agent.log import logger
from qwen_agent.utils.utils import print_traceback
from qwen_agent.gui.gradio import gr, mgr


class WebUI:
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
                yield _chatbot, _history
        if response:
            _history.extend([res for res in response if res[CONTENT] != PENDING_USER_INPUT])

        if _agent_selector is not None:
            yield _chatbot, _history, _agent_selector
        else:
            yield _chatbot, _history

        if self.verbose:
            logger.info('agent_run response:\n' + pprint.pformat(response, indent=2))

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
