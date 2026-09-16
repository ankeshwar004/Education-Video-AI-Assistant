from database.connection import pool


def create_user(username: str, normalized_username: str, password_hash: str):
    query = """
        INSERT INTO users (username, normalized_username, password_hash)
        VALUES (%s, %s, %s)
        RETURNING id, username, created_at;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (username, normalized_username, password_hash))
            user = cur.fetchone()
        conn.commit()
    return user


def get_user_by_username(normalized_username: str):
    query = """
        SELECT id, username, normalized_username, password_hash, created_at
        FROM users
        WHERE normalized_username = %s;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (normalized_username,))
            return cur.fetchone()


def get_user_by_id(user_id: int):
    query = """
        SELECT id, username, created_at
        FROM users
        WHERE id = %s;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id,))
            return cur.fetchone()