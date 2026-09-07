import uuid

from database.queries.videos_query import (
    create_video,
    get_video,
    get_videos,
    update_video_status,
    delete_video
)

from database.queries.chat_session_query import (
    create_chat_session,
    get_chat_session,
    get_chat_sessions_for_video,
    delete_chat_session,
)
from database.queries.chat_message_query import (
    create_chat_message,
    get_chat_messages,
)


from database.connection import pool


def test_connection():

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:

                cur.execute("SELECT version();")

                version = cur.fetchone()

                print("Connected successfully to Neon!")
                print(version)

    except Exception as e:

        print("Connection failed!")
        raise e


def test_database():

    print("\n========== DATABASE TEST START ==========\n")

    # --------------------------------
    # 1. CREATE VIDEO
    # --------------------------------

    video_id = f"test_video_{uuid.uuid4().hex[:8]}"

    print("1. Creating video...")

    video = create_video(
        video_id=video_id,
        title="Database Test Video",
        youtube_url="https://youtube.com/test",
        storage_path_url="/test/video.mp4",
        duration=120
    )

    print("Video created:")
    print(video)


    # --------------------------------
    # 2. GET VIDEO
    # --------------------------------

    print("\n2. Getting video...")

    fetched_video = get_video(video_id)

    assert fetched_video is not None

    print("Video found:")
    print(fetched_video)
    
    
    # --------------------------------
    # 2.5. UPDATE VIDEO STATUS
    # --------------------------------

    print("\n2.5. Updating video status...")

    update_video_status(video_id, status="processed")

    # --------------------------------
    # 3. CREATE CHAT SESSION
    # --------------------------------

    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    print("\n3. Creating chat session...")

    session = create_chat_session(
        session_id=session_id,
        video_id=video_id
    )

    assert session is not None

    print("Session created:")
    print(session)


    # --------------------------------
    # 4. GET CHAT SESSION
    # --------------------------------

    print("\n4. Getting chat session...")

    fetched_session = get_chat_session(session_id)

    assert fetched_session is not None

    print("Session found:")
    print(fetched_session)


    # --------------------------------
    # 5. GET SESSIONS FOR VIDEO
    # --------------------------------

    print("\n5. Getting sessions for video...")

    sessions = get_chat_sessions_for_video(video_id)

    assert len(sessions) > 0

    print("Sessions found:")
    print(sessions)


    # --------------------------------
    # 6. CREATE CHAT MESSAGES
    # --------------------------------

    print("\n6. Creating chat messages...")

    message_1 = create_chat_message(
        session_id=session_id,
        content="What is this video about?",
        role="user"
    )

    message_2 = create_chat_message(
        session_id=session_id,
        content="This is a database test video.",
        role="assistant"
    )

    assert message_1 is not None
    assert message_2 is not None

    print("Messages created:")
    print(message_1)
    print(message_2)


    # --------------------------------
    # 7. GET CHAT MESSAGES
    # --------------------------------

    print("\n7. Getting chat messages...")

    messages = get_chat_messages(session_id)

    assert len(messages) == 2

    print("Messages found:")
    print(messages)


    # --------------------------------
    # 8. DELETE CHAT SESSION
    # TEST CASCADE FOR MESSAGES
    # --------------------------------

    print("\n8. Deleting chat session...")

    delete_chat_session(session_id)

    deleted_session = get_chat_session(session_id)

    assert deleted_session is None

    deleted_messages = get_chat_messages(session_id)

    assert len(deleted_messages) == 0

    print("Chat session deleted successfully")
    print("Messages deleted automatically")


    # --------------------------------
    # 9. CREATE NEW SESSION
    # FOR VIDEO CASCADE TEST
    # --------------------------------

    print("\n9. Creating another session...")

    session_id_2 = f"test_session_{uuid.uuid4().hex[:8]}"

    create_chat_session(
        session_id=session_id_2,
        video_id=video_id
    )

    create_chat_message(
        session_id=session_id_2,
        content="Testing video cascade deletion",
        role="user"
    )


    # --------------------------------
    # 10. DELETE VIDEO
    # TEST CASCADE
    # --------------------------------

    print("\n10. Deleting video...")

    delete_video(video_id)

    deleted_video = get_video(video_id)

    assert deleted_video is None

    deleted_sessions = get_chat_sessions_for_video(video_id)

    assert len(deleted_sessions) == 0

    deleted_messages = get_chat_messages(session_id_2)

    assert len(deleted_messages) == 0

    print("Video deleted successfully")
    print("Sessions deleted automatically")
    print("Messages deleted automatically")


    print("\n========== ALL TESTS PASSED ==========\n")


if __name__ == "__main__":
    test_connection()
    test_database()