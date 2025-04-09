from websockets.sync.client import connect
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import os


STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Text, Voice="x_xiaomei"):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text
        self.Voice = Voice

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {
            "aue": "lame",
            "auf": "audio/L16;rate=16000",
            "vcn": self.Voice,
            "tte": "utf8",
            "sfl": 1
        }
        
        # 尝试不同的编码方式
        # 方式1：直接使用UTF-8编码
        text_bytes = self.Text.encode('utf-8')
        text_base64 = base64.b64encode(text_bytes).decode('utf-8')
        
        # 打印各种编码结果
        print("Original text:", self.Text)
        
        # 使用UTF-8编码方式，确保文本完整
        self.Data = {"status": 2, "text": text_base64}
        
        # 打印最终发送的数据
        # print("Final data text:", self.Data["text"])
        
        #使用小语种须使用以下方式，此处的unicode指的是 utf16小端的编码方式，即"UTF-16LE"
        # self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-16')), "UTF8")}

    # 生成url
    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        print('websocket url :', url)
        return url

def handle_message(message, output_path):
    """处理接收到的消息"""
    try:
        message = json.loads(message)
        code = message["code"]
        sid = message["sid"]
        audio = message["data"]["audio"]
        audio = base64.b64decode(audio)
        status = message["data"]["status"]
        print("Received message status:", status)
        
        if status == 2:
            print("ws is closed")
            return False
        if code != 0:
            errMsg = message["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
        else:
            with open(output_path, 'ab') as f:
                f.write(audio)
        return True
    except Exception as e:
        print("receive msg,but parse exception:", e)
        return False

def run_websocket(wsParam, output_path):
    """运行 WebSocket 客户端"""
    # 清理旧文件
    if os.path.exists(output_path):
        os.remove(output_path)

    # 创建 SSL 上下文
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    uri = wsParam.create_url()
    
    try:
        print(f"Connecting to {uri}")
        with connect(uri, ssl_context=ssl_context) as websocket:
            print("------>开始发送文本数据")
            
            # 准备发送数据
            d = {
                "common": wsParam.CommonArgs,
                "business": wsParam.BusinessArgs,
                "data": wsParam.Data,
            }
            
            # 打印发送的数据
            print("Sending data:", json.dumps(d, indent=2))
            
            # 发送数据
            websocket.send(json.dumps(d))
            
            # 接收响应
            while True:
                try:
                    message = websocket.recv()
                    if not handle_message(message, output_path):
                        break
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break
        return os.path.exists(output_path)
                
    except Exception as e:
        print(f"Error occurred: {e}")
    
