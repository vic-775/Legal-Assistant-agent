import os
from langfuse import Langfuse

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# Initialize Langfuse once
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
