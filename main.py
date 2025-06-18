import glob
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from jsonschema import validate
from typing_extensions import TypedDict


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    FastAPICache.init(InMemoryBackend())
    yield


class Login(TypedDict):
    date: str
    total: int


class User(TypedDict):
    id: str
    name: str


class Team(TypedDict):
    members: int
    leaders: int
    completed_projects: int
    active_percentage: float


class Country(TypedDict):
    country: str
    total: int


app = FastAPI(lifespan=lifespan)


@app.post("/users")
async def save_users(file: UploadFile | None = None) -> JSONResponse:
    if file:
        os.system("rm -rf db/*")
        content = await file.read()
        with open(f"db/{file.filename}", "wb") as f:
            f.write(content)

        df = pd.read_json(f"db/{file.filename}")
        return {"filename": file.filename, "user_count": df.shape[0]}
    else:
        return {"message": "ooops! Please upload a file"}


@cache(namespace="app", expire=20)
async def get_superusers(page: int, limit: int, *args, **kwargs) -> dict:
    offset = ((page - 1) * limit) if page > 1 else 0
    json_file = glob.glob("db/*.json")[0]
    df = pd.read_json(json_file)
    superusers = df[(df["score"] >= 900) & (df["active"] == True)][offset : (offset + limit)]  # noqa
    if "fields" in kwargs:
        superusers = superusers[kwargs.get("fields")]
    return superusers.to_dict(orient="records")


@app.get("/superusers")
async def superusers(limit: int = 50, page: int = 1) -> JSONResponse:
    start_time = time.perf_counter()
    response = await get_superusers(page, limit, fields=["id", "name"])
    process_time = time.perf_counter() - start_time
    return dict(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        execution_time_ms=process_time,
        data=response,
        page=page,
        offset=((page - 1) * limit) if page > 1 else 00,
        total=len(response),
    )


@app.get("/top-countries")
async def top_countries() -> JSONResponse:
    start_time = time.perf_counter()
    json_file = glob.glob("db/*.json")[0]
    df = pd.read_json(json_file)
    superusers = df[(df["score"] >= 900) & (df["active"] == True)]  # noqa
    response = (
        superusers.groupby("country", as_index=False).count().sort_values("id", ascending=False)[["country", "id"]][:5]
    )
    response.rename(columns={"id": "total"}, inplace=True)
    process_time = time.perf_counter() - start_time
    return dict(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        execution_time_ms=process_time,
        countries=response.to_dict(orient="records"),
    )


@app.get("/team-insights")
async def teams_insights() -> JSONResponse:
    start_time = time.perf_counter()
    json_file = glob.glob("db/*.json")[0]
    with open(json_file) as f:
        df2 = pd.json_normalize(
            json.load(f),
            meta=[
                "id",
                "name",
                "age",
                "score",
                ["team", "name"],
                ["team", "leader"],
                ["team", "projects", "name"],
                ["team", "projects", "completed"],
            ],
            max_level=2,
        )

    result = df2.groupby("team.name").agg(
        members=("name", "count"),
        leaders=("team.leader", "sum"),
        completed_projects=(
            "team.projects",
            lambda x: sum(prj["completed"] for p in x for prj in p),
        ),
        actives=("active", "sum"),
    )

    result["active_percentage"] = round((result["actives"] / result["members"]) * 100, 2)

    result.drop(columns=["actives"], inplace=True)

    process_time = time.perf_counter() - start_time
    return dict(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        execution_time_ms=process_time,
        teams=result.to_dict(orient="records"),
    )


@app.get("/active-users-per-day")
async def total_logins(min: int | None = None) -> JSONResponse:
    start_time = time.perf_counter()
    json_file = glob.glob("db/*.json")[0]
    with open(json_file) as f:
        df2 = pd.json_normalize(
            json.load(f),
            meta=[
                "id",
                "name",
                "age",
                "score",
                ["team", "name"],
                ["team", "leader"],
                ["team", "projects", "name"],
                ["team", "projects", "completed"],
            ],
            max_level=2,
        )

    logins = pd.DataFrame.from_records([obj for r in df2["logs"] for obj in r], columns=["date", "action"])

    response = logins.groupby("date", as_index=False).agg(
        total=("action", lambda x: sum(x == "login")),
    )[["date", "total"]]

    if min is not None:
        response = response[response["total"] >= min]

    process_time = time.perf_counter() - start_time

    return dict(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        execution_time_ms=process_time,
        logins=response.to_dict(orient="records"),
    )


@app.get("/evaluation")
async def health() -> JSONResponse:
    logins = await total_logins()
    teams = await teams_insights()
    countries = await top_countries()
    supers = await superusers()

    schema = {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string"},
            "execution_time_ms": {"type": "number"},
        },
    }
    login_schema = schema
    login_schema["properties"]["logins"] = {"type": "array"}
    teams_schema = schema
    teams_schema["properties"]["teams"] = {"type": "array"}
    countries_schema = schema
    countries_schema["properties"]["countries"] = {"type": "array"}
    super_schema = schema
    super_schema["properties"]["data"] = {"type": "array"}

    return {
        "endpoints": {
            "/active-users-per-day": dict(
                time_ms=logins["execution_time_ms"],
                valid_response=validate(instance=logins, schema=login_schema) is None,
            ),
            "/team_insights": dict(
                time_ms=teams["execution_time_ms"],
                valid_response=validate(instance=teams, schema=teams_schema) is None,
            ),
            "/top-countries": dict(
                time_ms=countries["execution_time_ms"],
                valid_response=validate(instance=countries, schema=countries_schema) is None,
            ),
            "/superusers": dict(
                time_ms=supers["execution_time_ms"],
                valid_response=validate(instance=supers, schema=super_schema) is None,
            ),
        }
    }
