from vanna.openai import OpenAI_Chat
from pinecone_vector_v2 import PineconeDB_VectorStore_V2
from openai import OpenAI
from langsmith.wrappers import wrap_openai


class CustomVanna(PineconeDB_VectorStore_V2, OpenAI_Chat):
    
    def __init__(self, config=None):
        if config is None:
            config = {}
            
        chat_api_key = config.get('llm_api_key')
        embedding_api_key = config.get('openai_api_key')
        llm_base_url = config.get('llm_base_url')
        
        if llm_base_url:
            self.chat_client = wrap_openai(OpenAI(api_key=chat_api_key, base_url=llm_base_url))
        else:
            self.chat_client = wrap_openai(OpenAI(api_key=embedding_api_key))
        self.embedding_client = wrap_openai(OpenAI(api_key=embedding_api_key, base_url="https://api.openai.com/v1"))

        pinecone_config = {
            'api_key': config.get('pinecone_api_key'),
            'dimensions': 1536,
            'embedding_model': config.get('embedding_model'),
            'embedding_client': self.embedding_client,
            'n_results': 3
        }
        
        PineconeDB_VectorStore_V2.__init__(self, config=pinecone_config)
        OpenAI_Chat.__init__(self, 
            client=self.chat_client, 
            config={
                'model': config.get('openai_model', 'gpt-4'),
                'allow_llm_to_see_data': True
            })
        
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
        