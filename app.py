from dotenv import load_dotenv
load_dotenv()

import os
from cache import MemoryCache
from custom_vanna import CustomVanna
from vanna.flask import VannaFlaskApp
from auth import SimplePassword
from custom_vanna import CustomVannaFlaskApp

vn = CustomVanna(config={
    'llm_base_url': os.environ['LLM_BASE_URL'],
    'llm_api_key': os.environ['LLM_API_KEY'],
    'llm_model': os.environ['LLM_MODEL'],
    'embedding_model': os.environ['EMBEDDING_MODEL'],
    'embedding_api_key': os.environ['EMBEDDING_API_KEY']
});

# Connect to database
vn.connect_to_database(
    db_type=os.environ.get('DB_TYPE', 'mysql'),
    host=os.environ['DB_HOST'],
    dbname=os.environ['DB_NAME'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASSWORD'],
    port=os.environ['DB_PORT']
)
current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
  CustomVannaFlaskApp(
        vn=vn,
        # auth=SimplePassword(users=[{"email": os.environ['USER_EMAIL'], "password": os.environ['USER_PASSWORD']}]),
        cache=MemoryCache(),
        allow_llm_to_see_data=True,
        logo="https://cslrvbcjymwwcpwfzbuh.supabase.co/storage/v1/object/public/images//cherrypicks_logo_TC_portrait_png.svg",
        title="AI SQL Generation",
        subtitle="Your AI-powered copilot for SQL queries.",
        show_training_data=True,
        sql=True,
        table=True,
        chart=False,
        summarization=False,
        ask_results_correct=False,
        debug=True,
        index_html_path=os.path.join(current_dir, "static/index.html"),
        assets_folder=os.path.join(current_dir, "static/assets")
    ).run()
