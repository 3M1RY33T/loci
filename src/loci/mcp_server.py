"""MCP server: three tools, on purpose.

Agents are the primary consumer of a memory tool, so this is the surface that
matters most. It is also where memory tools most often go wrong: exposing
several dozen fine-grained tools pushes the orchestration burden onto the model
and burns a turn per hop. loci exposes the decisions an agent actually needs:

    ask      route a question and answer it in scope
    scopes   what is indexed
    doctor   where coverage is missing, and what to run to fix it

`ask` takes `cwd` because routing is dramatically better with it -- questions
that name no project ("how do I run the tests?") route correctly with the
working directory and barely at all without. An MCP client that knows the user's
directory should always pass it.
"""
from __future__ import annotations

import json
import sys


def _text(s: str):
    from mcp import types
    return [types.TextContent(type="text", text=s)]


def build_server():
    from mcp import types
    from mcp.server import Server

    server = Server("loci")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="ask",
                description=(
                    "Answer a question about the user's projects. Routes to the "
                    "right project automatically, then searches both its code "
                    "structure and its written history (notes, docs, commits). "
                    "Returns ABSTAINED when the question is too vague to route -- "
                    "ask the user which project rather than guessing."),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "cwd": {"type": "string", "description":
                                "The user's working directory. Pass it whenever "
                                "known: it is the strongest routing signal."},
                        "scope": {"type": "string", "description":
                                  "Force a project by name, bypassing routing."},
                        "group": {"type": "string", "description":
                                  "Restrict to one group of projects, e.g. "
                                  "'me' or 'client:delroy'."},
                        "k": {"type": "integer", "default": 3,
                              "description": "Episode hits per scope."},
                        "rerank": {"type": "boolean", "default": False,
                                   "description": "Cross-encoder rerank; slower, "
                                                  "better ordering."},
                    },
                    "required": ["question"],
                },
            ),
            types.Tool(
                name="scopes",
                description="List indexed projects with their sizes.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="doctor",
                description=(
                    "Report which projects have missing or thin coverage, and the "
                    "command to fix each. Use when a question returns no evidence."),
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None):
        args = arguments or {}
        try:
            from .index import load_episodes, load_index
        except Exception as exc:
            return _text(f"loci is not set up: {exc}")

        try:
            index = load_index()
        except (FileNotFoundError, ValueError) as exc:
            return _text(f"{exc}\nRun `loci scan <dir>` then `loci index`.")

        if name == "scopes":
            rows = [f"{m['name']}  ({m.get('structure_nodes', 0)} symbols, "
                    f"{m.get('chunk_count', 0)} chunks)"
                    for m in index["scopes"].values()]
            return _text("\n".join(rows) or "no scopes indexed")

        if name == "doctor":
            from .doctor import check, render
            from .scopes import load_scopes
            return _text(render(check(index, load_episodes(), load_scopes())))

        if name == "ask":
            question = (args.get("question") or "").strip()
            if not question:
                return _text("question is required")
            from .ask import ask, render
            forced = None
            if args.get("scope"):
                by_name = {m["name"].lower(): s for s, m in index["scopes"].items()}
                want = str(args["scope"])
                sid = want if want in index["scopes"] else by_name.get(want.lower())
                if not sid:
                    return _text(f"unknown scope {want!r}; have: "
                                 f"{', '.join(m['name'] for m in index['scopes'].values())}")
                forced = [sid]
            answer = ask(question, cwd=args.get("cwd"), episodes_k=int(args.get("k", 3)),
                         rerank=bool(args.get("rerank", False)),
                         force_scopes=forced, group=args.get("group"),
                         index=index, store=load_episodes())
            return _text(render(answer, index=index))

        return _text(f"unknown tool: {name}")

    return server


def _warm_up() -> None:
    """Load the embedding model and the largest scopes' rankers at boot.

    Without this the FIRST tool call an agent makes pays ~4.4s of import and
    model construction while the user waits, and every call after it costs
    ~0.1s. A server that is going to pay that cost anyway should pay it before
    anyone is watching.
    """
    try:
        from .backends import episodes as ep
        from .index import load_episodes, load_index
        index = load_index()
        store = load_episodes()
        biggest = sorted(index["scopes"],
                         key=lambda s: -index["scopes"][s].get("chunk_count", 0))[:3]
        ep.warm_up(store, biggest)
    except Exception:
        pass  # a cold server still answers; it is just slower on the first call


def main() -> int:
    import anyio
    from mcp.server.stdio import stdio_server

    async def run() -> None:
        _warm_up()
        server = build_server()
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        anyio.run(run)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
