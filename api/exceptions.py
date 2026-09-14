from fastapi import HTTPException


def not_found(detail: str):
    raise HTTPException(status_code=404, detail=detail)


def bad_request(detail: str):
    raise HTTPException(status_code=400, detail=detail)


def conflict(detail: str):
    raise HTTPException(status_code=409, detail=detail)


def unprocessable(detail: str):
    raise HTTPException(status_code=422, detail=detail)