from fastapi import HTTPException


def not_found(detail: str):
    return HTTPException(status_code=404, detail=detail)


def bad_request(detail: str):
    return HTTPException(status_code=400, detail=detail)


def conflict(detail: str):
    return HTTPException(status_code=409, detail=detail)


def unprocessable(detail: str):
    return HTTPException(status_code=422, detail=detail)