from src.chat import lcel_chat
from src.memory import initialize_memory

from services.retrieval_service import load_retrieval_components


def ask_question(question,video_id,session_id):

    retrieval_components = load_retrieval_components(video_id)

    initialize_memory(session_id,video_id)

    response = lcel_chat(question,retrieval_components,session_id)

    return response