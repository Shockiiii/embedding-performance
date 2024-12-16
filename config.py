import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default configurations
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"
TEXT_FILE_PATH = "example.txt"
TOP_K = 3  # Number of similar chunks to retrieve
