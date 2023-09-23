
import base64
import hmac
import json
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse
from websocket import create_connection, WebSocketConnectionClosedException
from utils.tools import get_prompt, process_response, init_script, create_script


class SparkAPI:
    __api_url = 'ws://spark-api.xf-yun.com/v2.1/chat'#ws://spark-api.xf-yun.com/v2.1/chat v2的地址 wss://spark-api.xf-yun.com/v1.1/chat
    __max_token = 4096

    def __init__(self):
        self.__app_id = "9791fb23"
        self.__api_key = "9c7969e87a3766da0bc4f5edd9607065"
        self.__api_secret = "M2NiMjZkYWZkZDI3OWVmN2ExN2FlMDlh"

    def __set_max_tokens(self, token):
        if isinstance(token, int) is False or token < 0:
            print("set_max_tokens() error: tokens should be a positive integer!")
            return
        self.__max_token = token

    def __get_authorization_url(self):
        authorize_url = urlparse(self.__api_url)
        # 1. generate data
        date = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S %Z')

        """
        Generation rule of Authorization parameters
            1) Obtain the APIKey and APISecret parameters from the console.
            2) Use the aforementioned date to dynamically concatenate a string tmp. Here we take Huobi's URL as an example, 
                the actual usage requires replacing the host and path with the specific request URL.
        """
        signature_origin = "host: {}\ndate: {}\nGET {} HTTP/1.1".format(
            authorize_url.netloc, date, authorize_url.path
        )
        signature = base64.b64encode(
            hmac.new(
                self.__api_secret.encode(),
                signature_origin.encode(),
                digestmod='sha256'
            ).digest()
        ).decode()
        authorization_origin = \
            'api_key="{}",algorithm="{}",headers="{}",signature="{}"'.format(
                self.__api_key, "hmac-sha256", "host date request-line", signature
            )
        authorization = base64.b64encode(
            authorization_origin.encode()).decode()
        params = {
            "authorization": authorization,
            "date": date,
            "host": authorize_url.netloc
        }

        ws_url = self.__api_url + "?" + urlencode(params)
        return ws_url

    def __build_inputs(
            self,
            message: dict,
            user_id: str = "001",
            domain: str = "generalv2",
            temperature: float = 0.5,
            max_tokens: int = 4096
    ):
        input_dict = {
            "header": {
                "app_id": self.__app_id,
                "uid": user_id,
            },
            "parameter": {
                "chat": {
                    "domain": domain,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            },
            "payload": {
                "message": message
            }
        }
        return json.dumps(input_dict)

    def chat(
            self,
            query: str,
            history: list = None,  # store the conversation history
            user_id: str = "001",
            domain: str = "generalv2",
            max_tokens: int = 4096,
            temperature: float = 0.5,
    ):
        if history is None:
            history = []

        # the max of max_length is 4096
        max_tokens = min(max_tokens, 4096)
        url = self.__get_authorization_url()
        ws = create_connection(url)
        message = get_prompt(query, history)
        input_str = self.__build_inputs(
            message=message,
            user_id=user_id,
            domain=domain,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ws.send(input_str)
        response_str = ws.recv()
        try:
            while True:
                response, history, status = process_response(
                    response_str, history)
                """
                The final return result, which means a complete conversation.
                doc url: https://www.xfyun.cn/doc/spark/Web.html#_1-%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E
                """
                if len(response) == 0 or status == 2:
                    break
                response_str = ws.recv()
            return response

        except WebSocketConnectionClosedException:
            print("Connection closed")
        finally:
            ws.close()
    # Stream output statement, used for terminal chat.

    def streaming_output(
            self,
            query: str,
            history: list = None,  # store the conversation history
            user_id: str = "001",
            domain: str = "generalv2",
            max_tokens: int = 4096,
            temperature: float = 0.5,
    ):
        if history is None:
            history = []
        # the max of max_length is 4096
        max_tokens = min(max_tokens, 4096)
        url = self.__get_authorization_url()
        ws = create_connection(url)

        message = get_prompt(query, history)
        input_str = self.__build_inputs(
            message=message,
            user_id=user_id,
            domain=domain,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # print(input_str)
        # send question or prompt to url, and receive the answer
        ws.send(input_str)
        response_str = ws.recv()

        # Continuous conversation
        try:
            while True:
                response, history, status = process_response(
                    response_str, history)
                yield response, history
                if len(response) == 0 or status == 2:
                    break
                response_str = ws.recv()

        except WebSocketConnectionClosedException:
            print("Connection closed")
        finally:
            ws.close()

    def chat_stream(self):
        history = []
        try:
            while True:
                query = input("Ask: ")
                for response, _ in self.streaming_output(query, history):
                    print("\r" + response, end="")
                print("\n")
        finally:
            print("\nThank you for using the SparkDesk AI. Welcome to use it again!")

class Spark_toolsAPI:
    __api_url = 'wss://spark-openapi.cn-huabei-1.xf-yun.com/v1/assistants/u2hirwydfmrs_v1'#ws://spark-api.xf-yun.com/v2.1/chat v2的地址 wss://spark-api.xf-yun.com/v1.1/chat
    __max_token = 4096

    def __init__(self):
        self.__app_id = "9791fb23"
        self.__api_key = "9c7969e87a3766da0bc4f5edd9607065"
        self.__api_secret = "M2NiMjZkYWZkZDI3OWVmN2ExN2FlMDlh"

    def __set_max_tokens(self, token):
        if isinstance(token, int) is False or token < 0:
            print("set_max_tokens() error: tokens should be a positive integer!")
            return
        self.__max_token = token

    def __get_authorization_url(self):
        authorize_url = urlparse(self.__api_url)
        # 1. generate data
        date = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S %Z')

        """
        Generation rule of Authorization parameters
            1) Obtain the APIKey and APISecret parameters from the console.
            2) Use the aforementioned date to dynamically concatenate a string tmp. Here we take Huobi's URL as an example, 
                the actual usage requires replacing the host and path with the specific request URL.
        """
        signature_origin = "host: {}\ndate: {}\nGET {} HTTP/1.1".format(
            authorize_url.netloc, date, authorize_url.path
        )
        signature = base64.b64encode(
            hmac.new(
                self.__api_secret.encode(),
                signature_origin.encode(),
                digestmod='sha256'
            ).digest()
        ).decode()
        authorization_origin = \
            'api_key="{}",algorithm="{}",headers="{}",signature="{}"'.format(
                self.__api_key, "hmac-sha256", "host date request-line", signature
            )
        authorization = base64.b64encode(
            authorization_origin.encode()).decode()
        params = {
            "authorization": authorization,
            "date": date,
            "host": authorize_url.netloc
        }

        ws_url = self.__api_url + "?" + urlencode(params)
        return ws_url

    def __build_inputs(
            self,
            message: dict,
            user_id: str = "001",
            domain: str = "generalv2",
            temperature: float = 0.5,
            max_tokens: int = 4096
    ):
        input_dict = {
            "header": {
                "app_id": self.__app_id,
                "uid": user_id,
            },
            "parameter": {
                "chat": {
                    "domain": domain,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            },
            "payload": {
                "message": message
            }
        }
        return json.dumps(input_dict)

    def chat(
            self,
            query: str,
            history: list = None,  # store the conversation history
            user_id: str = "001",
            domain: str = "generalv2",
            max_tokens: int = 4096,
            temperature: float = 0.5,
    ):
        if history is None:
            history = []

        # the max of max_length is 4096
        max_tokens = min(max_tokens, 4096)
        url = self.__get_authorization_url()
        ws = create_connection(url)
        message = get_prompt(query, history)
        input_str = self.__build_inputs(
            message=message,
            user_id=user_id,
            domain=domain,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ws.send(input_str)
        response_str = ws.recv()
        try:
            while True:
                response, history, status = process_response(
                    response_str, history)
                """
                The final return result, which means a complete conversation.
                doc url: https://www.xfyun.cn/doc/spark/Web.html#_1-%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E
                """
                if len(response) == 0 or status == 2:
                    break
                response_str = ws.recv()
            return response

        except WebSocketConnectionClosedException:
            print("Connection closed")
        finally:
            ws.close()
    # Stream output statement, used for terminal chat.

    def streaming_output(
            self,
            query: str,
            history: list = None,  # store the conversation history
            user_id: str = "001",
            domain: str = "generalv2",
            max_tokens: int = 4096,
            temperature: float = 0.5,
    ):
        if history is None:
            history = []
        # the max of max_length is 4096
        max_tokens = min(max_tokens, 4096)
        url = self.__get_authorization_url()
        ws = create_connection(url)

        message = get_prompt(query, history)
        input_str = self.__build_inputs(
            message=message,
            user_id=user_id,
            domain=domain,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # print(input_str)
        # send question or prompt to url, and receive the answer
        ws.send(input_str)
        response_str = ws.recv()

        # Continuous conversation
        try:
            while True:
                response, history, status = process_response(
                    response_str, history)
                yield response, history
                if len(response) == 0 or status == 2:
                    break
                response_str = ws.recv()

        except WebSocketConnectionClosedException:
            print("Connection closed")
        finally:
            ws.close()

    def chat_stream(self):
        history = []
        try:
            while True:
                query = input("Ask: ")
                for response, _ in self.streaming_output(query, history):
                    print("\r" + response, end="")
                print("\n")
        finally:
            print("\nThank you for using the SparkDesk AI. Welcome to use it again!")

from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
class Spark_forlangchain(LLM):

    # 类的成员变量，类型为整型
    n: int
    # 用于指定该子类对象的类型

    @property
    def _llm_type(self) -> str:
        return "Spark"

    # 重写基类方法，根据用户输入的prompt来响应用户，返回字符串
    def _call(
            self,
            query: str,
            history: list = None,  # store the conversation history
            user_id: str = "001",
            domain: str = "generalv2",
            max_tokens: int = 4096,
            temperature: float = 0.7,
            stop: Optional[List[str]] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        bot = SparkAPI()
        response = bot.chat(query, history, user_id,
                            domain, max_tokens, temperature)
        return response

    # 返回一个字典类型，包含LLM的唯一标识
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
class Spark_tools_forlangchain(LLM):

    # 类的成员变量，类型为整型
    n: int
    # 用于指定该子类对象的类型

    @property
    def _llm_type(self) -> str:
        return "Spark"

    # 重写基类方法，根据用户输入的prompt来响应用户，返回字符串
    def _call(
            self,
            query: str,
            history: list = None,  # store the conversation history
            user_id: str = "001",
            domain: str = "generalv2",
            max_tokens: int = 4096,
            temperature: float = 0.7,
            stop: Optional[List[str]] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        bot = Spark_toolsAPI()
        response = bot.chat(query, history, user_id,
                            domain, max_tokens, temperature)
        return response

    # 返回一个字典类型，包含LLM的唯一标识
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}