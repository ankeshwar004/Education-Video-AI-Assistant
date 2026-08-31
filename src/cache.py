import os
import json
import hashlib
import redis
import config

redis_client = redis.Redis(host=config.REDIS_HOST,port=int(config.REDIS_PORT),decode_responses=True)

def make_key(prefix, *values):
    raw = "||".join(str(v) for v in values)
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return f"{prefix}:{hashed}"


def get_cache(prefix, *values):
    key = make_key(prefix, *values)
    value = redis_client.get(key)

    if value is None:
        return None

    return json.loads(value)


def set_cache(prefix, value, ttl, *values):
    key = make_key(prefix, *values)
    redis_client.setex(key,ttl,json.dumps(value))
    


def make_messages_hash(messages):

    serialized = [{"type": message.type,"content": message.content } for message in messages]

    data = json.dumps(
        serialized,
        sort_keys=True,
        ensure_ascii=False,
        default=str
    )

    return hashlib.sha256(data.encode()).hexdigest()