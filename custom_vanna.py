from vanna.openai import OpenAI_Chat
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from vanna.chromadb import ChromaDB_VectorStore
from chromadb.api.types import EmbeddingFunction
from vanna.flask import VannaFlaskAPI, VannaFlaskApp
from flask import jsonify, request, send_from_directory, url_for
from vanna.flask.auth import NoAuth
import os
from cache import MemoryCache
import chromadb
from chromadb.config import Settings

from xunfei_tts_ws_python3 import Ws_Param, run_websocket
import uuid
class OpenAIEmbedding(EmbeddingFunction):
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002", base_url: str = "https://api.openai.com/v1"):
        self.client = wrap_openai(OpenAI(api_key=api_key, base_url=base_url))
        self.model = model

    def __call__(self, texts):
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            # Handle empty input
            if not texts:
                return []
                
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            # Extract embeddings and ensure they're in the correct format
            embeddings = [data.embedding for data in response.data]
            return embeddings
            
        except Exception as e:
            print(f"Error in OpenAIEmbedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
        

class CustomVanna(ChromaDB_VectorStore, OpenAI_Chat):
    
    def __init__(self, config=None):
        if config is None:
            config = {}
            
        chat_api_key = config.get('llm_api_key')
        embedding_api_key = config.get('embedding_api_key')
        llm_base_url = config.get('llm_base_url')
        llm_model = config.get('llm_model')
        self.embedding_model = config.get('embedding_model')
        self.chat_client = wrap_openai(OpenAI(api_key=chat_api_key, base_url=llm_base_url))
        self.embedding_client = wrap_openai(OpenAI(api_key=embedding_api_key, base_url="https://api.openai.com/v1"))
        
        chroma_config = {
            'path': config.get('chroma_path', './chroma'),
            'client': 'persistent',
            'embedding_function': OpenAIEmbedding(api_key=embedding_api_key, model=self.embedding_model),
            'n_results_sql': 20,
            'n_results_documentation': 5,
            'n_results_ddl': 5,
        }
        
        ChromaDB_VectorStore.__init__(self, config=chroma_config)
        OpenAI_Chat.__init__(self, 
            client=self.chat_client, 
            config={
                'model': llm_model,
                'allow_llm_to_see_data': True
            })
        
    def generate_embedding(self, data: str, **kwargs) -> list:
        try:
            response = self.embedding_client.embeddings.create(
                input=data,
                model=self.embedding_model 
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise e
        
    def connect_to_database(self, db_type='mysql', **kwargs):
        """
        Connect to database based on type
        Args:
            db_type: 'mysql' or 'postgresql'
            **kwargs: connection parameters (host, dbname, user, password, port)
        """
        if db_type.lower() == 'mysql':
            self.connect_to_mysql(
                host=kwargs.get('host'),
                dbname=kwargs.get('dbname'),
                user=kwargs.get('user'),
                password=kwargs.get('password'),
                port=int(kwargs.get('port', 3306))
            )
        elif db_type.lower() in ['postgresql', 'postgres']:
            self.connect_to_postgres(
                host=kwargs.get('host'),
                dbname=kwargs.get('dbname'),
                user=kwargs.get('user'),
                password=kwargs.get('password'),
                port=int(kwargs.get('port', 5432))
            )

class CustomVannaFlaskApp(VannaFlaskApp):
    def __init__(
        self,
        vn,
        cache=MemoryCache(),
        auth=NoAuth(),
        debug=True,
        allow_llm_to_see_data=False,
        logo="https://img.vanna.ai/vanna-flask.svg",
        title="Welcome to Vanna.AI",
        subtitle="Your AI-powered copilot for SQL queries.",
        show_training_data=True,
        suggested_questions=True,
        sql=True,
        table=True,
        csv_download=True,
        chart=True,
        redraw_chart=True,
        auto_fix_sql=True,
        ask_results_correct=True,
        followup_questions=True,
        summarization=True,
        function_generation=True,
        index_html_path=None,
        assets_folder=None,
    ):
        # Initialize VannaFlaskApp first
        super().__init__(
            vn=vn,
            cache=cache,
            auth=auth,
            debug=debug,
            allow_llm_to_see_data=allow_llm_to_see_data,
            logo=logo,
            title=title,
            subtitle=subtitle,
            show_training_data=show_training_data,
            suggested_questions=suggested_questions,
            sql=sql,
            table=table,
            csv_download=csv_download,
            chart=chart,
            redraw_chart=redraw_chart,
            auto_fix_sql=auto_fix_sql,
            ask_results_correct=ask_results_correct,
            followup_questions=followup_questions,
            summarization=summarization,
            function_generation=function_generation,
            index_html_path=index_html_path,
            assets_folder=assets_folder,
        )
        
        # Remove the existing route
        if 'get_training_data' in self.flask_app.view_functions:
            del self.flask_app.view_functions['get_training_data']
        
        # Remove the existing url rule
        for rule in self.flask_app.url_map.iter_rules():
            if rule.endpoint == 'get_training_data':
                self.flask_app.url_map._rules.remove(rule)
                self.flask_app.url_map._rules_by_endpoint['get_training_data'].remove(rule)
                if not self.flask_app.url_map._rules_by_endpoint['get_training_data']:
                    del self.flask_app.url_map._rules_by_endpoint['get_training_data']
                break
        
        # Add our custom endpoint
        @self.flask_app.route("/api/v0/get_training_data", methods=["GET"])
        @self.requires_auth
        def get_training_data(user: any):
            try:
                df = self.vn.get_training_data()
                
                # if df is None or len(df) == 0:
                #     return jsonify({
                #         "type": "error",
                #         "error": "No training data found. Please add some training data first."
                #     })

                return jsonify({
                    "type": "df",
                    "id": "training_data",
                    "df": df.to_json(orient="records")
                })
                
            except Exception as e:
                return jsonify({
                    "type": "error",
                    "error": str(e)
                })
                
                
        @self.flask_app.route("/static/audio/<path:filename>")
        def proxy_audio(filename):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            audio_dir = os.path.join(current_dir, 'static', 'audio')
            if filename:
              return send_from_directory(audio_dir, filename)

            # Return 404
            return "File not found", 404
                
        @self.flask_app.route("/api/v0/generate_tts", methods=["POST"])
        @self.requires_auth
        def generate_tts(user: any):
            try:
                # 从请求中获取所有参数
                request_data = request.get_json()
                
                # 检查必需的参数
                required_params = ['text', 'appId', 'apiSecret', 'apiKey']
                missing_params = [param for param in required_params if param not in request_data]
                
                if missing_params:
                    return jsonify({
                        "type": "error",
                        "error": f"Missing required parameters: {', '.join(missing_params)}"
                    }), 400
                    
                # 获取参数
                text = request_data['text']
                app_id = request_data['appId']
                api_secret = request_data['apiSecret']
                api_key = request_data['apiKey']
                voice = request_data.get('voice', 'x_xiaomei')
                
                # 生成唯一的文件名
                output_filename = f'tts_{uuid.uuid4()}.mp3'
                audio_dir = os.path.join('./static', 'audio')
                os.makedirs(audio_dir, exist_ok=True)
                output_path = os.path.join(audio_dir, output_filename)
                
                # 初始化讯飞TTS参数
                wsParam = Ws_Param(
                    APPID=app_id,
                    APISecret=api_secret,
                    APIKey=api_key,
                    Text=text,
                    Voice=voice
                )

                # 运行TTS转换
                success = run_websocket(wsParam, output_path)
                
                if not success:
                    return jsonify({
                        "type": "error",
                        "error": "Failed to generate audio"
                    }), 500

                # 返回音频文件URL
                relative_audio_url = url_for('static', filename=f'audio/{output_filename}')
        
                # 从环境变量中获取域名
                domain = os.environ.get("DOMAIN", "http://localhost:80")
                
                # 拼接完整的 URL
                audio_url = f"{domain}{relative_audio_url}"
                        
                return jsonify({
                    "type": "success",
                    "data": {
                        "audio_url": audio_url,
                        "text": text,
                        "voice": voice
                    }
                })
                
            except Exception as e:
                return jsonify({
                    "type": "error",
                    "error": str(e)
                })
        
        