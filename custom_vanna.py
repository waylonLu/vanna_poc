from vanna.openai import OpenAI_Chat
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from vanna.chromadb import ChromaDB_VectorStore
from chromadb.api.types import EmbeddingFunction
from vanna.flask import VannaFlaskAPI, VannaFlaskApp
from flask import jsonify
from vanna.flask.auth import NoAuth
import os
from cache import MemoryCache
import chromadb
from chromadb.config import Settings
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
        self.chroma_client = chromadb.HttpClient(
            host=os.environ['CHROMA_HOST'], 
            port=os.environ['CHROMA_PORT'], 
            settings=Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=True, 
                              chroma_server_host=os.environ['CHROMA_HOST'], chroma_server_http_port=int(os.environ['CHROMA_PORT']))
        )
        
        chroma_config = {
            'path': config.get('chroma_path', './chroma'),
            'client': self.chroma_client,
            'embedding_function': OpenAIEmbedding(api_key=embedding_api_key, model=self.embedding_model),
            'n_results_sql': 5,
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
        
        