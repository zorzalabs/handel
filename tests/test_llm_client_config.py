import json
from pathlib import Path

import httpx
import pytest

from handel.config import build_agents, load_config, parse_config, request_params
from handel.engine.state import GameConfig
from handel.llm import LLMAgent, LLMError, OpenRouterClient, OptionPickerClient, Price
from handel.llm.client import OPENROUTER_URL
from handel.play import play_one

ROOT = Path(__file__).resolve().parents[1]


def _ok(content='{"action": "buy"}', usage=None):
    return httpx.Response(200, json={
        "id": "gen-1", "model": "vendor/m",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage if usage is not None else {
            "prompt_tokens": 1000, "completion_tokens": 50, "cost": 0.0123,
            "prompt_tokens_details": {"cached_tokens": 800},
            "completion_tokens_details": {"reasoning_tokens": 20}},
    })


def client_with(responses, **kw):
    seen = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    c = OpenRouterClient(api_key="test-key", transport=httpx.MockTransport(handler),
                         sleep=lambda _: None, **kw)
    return c, seen


def test_openrouter_request_and_usage_parsing():
    c, seen = client_with([_ok()])
    out = c.complete("vendor/m", [{"role": "user", "content": "hi"}], temperature=0.2,
                     response_format={"type": "json_object"})
    req = seen[0]
    assert str(req.url) == OPENROUTER_URL
    assert req.headers["Authorization"] == "Bearer test-key"
    body = json.loads(req.content)
    assert body["model"] == "vendor/m" and body["temperature"] == 0.2
    assert body["response_format"] == {"type": "json_object"}
    assert (out.text, out.prompt_tokens, out.completion_tokens) == ('{"action": "buy"}', 1000, 50)
    assert out.cost_usd == 0.0123 and out.cached_tokens == 800 and out.reasoning_tokens == 20


def test_openrouter_retries_transient_errors():
    c, seen = client_with([httpx.Response(429, text="slow down"),
                           httpx.ConnectError("boom"),
                           httpx.Response(503, text="busy"),
                           _ok()])
    assert c.complete("vendor/m", []).text == '{"action": "buy"}'
    assert len(seen) == 4


def test_openrouter_gives_up_and_raises():
    c, _ = client_with([httpx.Response(500)] * 3, max_retries=2)
    with pytest.raises(LLMError, match="gave up after 3"):
        c.complete("vendor/m", [])
    c2, seen = client_with([httpx.Response(400, text="bad model")])
    with pytest.raises(LLMError, match="HTTP 400"):
        c2.complete("vendor/m", [])
    assert len(seen) == 1


def test_cost_estimated_from_pricing_when_missing():
    c, _ = client_with([_ok(usage={"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000})],
                       pricing={"vendor/m": Price(0.5, 2.0)})
    assert c.complete("vendor/m", []).cost_usd == pytest.approx(3.0)


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        OpenRouterClient()


def test_example_configs_parse():
    demo = load_config(ROOT / "configs/offline_demo.toml")
    assert [s.kind for s in demo.seats] == ["llm", "llm", "bot", "bot"]
    assert not demo.needs_api
    real = load_config(ROOT / "configs/openrouter_example.toml")
    assert real.needs_api and real.usd_to_eur == 0.92
    assert real.seats[1].llm["temperature"] == 0.3 and real.seats[0].llm["temperature"] == 0.7
    assert request_params(real.seats[0].llm) == {
        "temperature": 0.7, "max_tokens": 800, "response_format": {"type": "json_object"}}


@pytest.mark.parametrize("raw, fragment", [
    ({"seats": [{"type": "bot", "bot": "genius"}] * 2}, "unknown bot"),
    ({"seats": [{"type": "llm"}] * 2}, "need a model"),
    ({"seats": [{"type": "llm", "model": "a/b", "temprature": 1}] * 2}, "unknown keys"),
    ({"seats": [{"type": "human"}] * 2}, "type must be"),
    ({"seats": [{"type": "bot", "bot": "random"}]}, "2 to 6"),
    ({"game": {"rounds": 5}, "seats": [{"type": "bot", "bot": "random"}] * 2}, "unknown [game]"),
])
def test_config_errors(raw, fragment):
    with pytest.raises(ValueError) as e:
        parse_config(raw)
    assert fragment in str(e.value)


def test_build_agents_uses_shared_client_for_real_models():
    cfg = parse_config({"llm": {"extra": {"reasoning": {"effort": "low"}}},
                        "seats": [{"type": "llm", "model": "a/b"}, {"type": "llm", "model": "c/d"},
                                  {"type": "bot", "bot": "heuristic"}]})
    sentinel = object()
    agents = build_agents(cfg, seed=1, client=sentinel)  # type: ignore[arg-type]
    assert agents[0].client is sentinel and agents[1].client is sentinel
    assert agents[0].params["reasoning"] == {"effort": "low"}
    assert not isinstance(agents[2], LLMAgent)


def test_play_one_writes_a_complete_record(tmp_path):
    cfg = load_config(ROOT / "configs/offline_demo.toml")
    result = play_one(cfg, seed=2, out_dir=tmp_path)
    rec = json.loads((tmp_path / "game_2.json").read_text())
    assert rec["result"]["winner"] == result.winner
    assert set(rec["transcripts"]) == {"0", "1"}
    assert rec["transcripts"]["0"][0]["prompt"].startswith("ROUND")
    assert rec["log"][-1]["ev"] == "game_over"
    assert "cost_eur" in rec and rec["system_prompt"].startswith("You are playing HANDEL")


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_and_billing_errors_are_fatal_and_not_retried(status):
    from handel.llm import LLMFatalError

    body = '{"error":{"message":"User not found.","code":401}}'
    c, seen = client_with([httpx.Response(status, text=body)] * 3)
    with pytest.raises(LLMFatalError, match=f"HTTP {status}"):
        c.complete("vendor/m", [])
    assert len(seen) == 1


def test_play_stops_with_clear_message_on_bad_key(tmp_path, monkeypatch):
    from handel import play
    from handel.llm import LLMFatalError

    def fail(*a, **k):
        raise LLMFatalError('HTTP 401: {"error":{"message":"User not found."}}')

    monkeypatch.setattr(play, "play_one", fail)
    with pytest.raises(SystemExit) as e:
        play.main([str(ROOT / "configs/claude_trio.toml"), "--out", str(tmp_path)])
    assert "OPENROUTER_API_KEY" in str(e.value) and "401" in str(e.value)


def test_dotenv_loading(tmp_path, monkeypatch):
    from handel.play import load_dotenv

    monkeypatch.setenv("OPENROUTER_API_KEY", "stale-shell-value")
    monkeypatch.delenv("HANDEL_EMPTY", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nOPENROUTER_API_KEY="sk-or-from-file"\nHANDEL_EMPTY=\nnot a pair\n')
    assert load_dotenv(env) == ["OPENROUTER_API_KEY"]
    import os

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-file"  # file wins over the shell
    assert "HANDEL_EMPTY" not in os.environ  # empty values are ignored
    assert load_dotenv(tmp_path / "missing.env") == []


def test_cache_write_tokens_parsed():
    c, _ = client_with([_ok(usage={"prompt_tokens": 3000, "completion_tokens": 10, "cost": 0.01,
                                   "prompt_tokens_details": {"cached_tokens": 0,
                                                             "cache_write_tokens": 1500}})])
    out = c.complete("vendor/m", [])
    assert out.cache_write_tokens == 1500 and out.cached_tokens == 0


def test_cache_prompt_auto_for_anthropic_models():
    from handel.config import cache_prompt_enabled

    assert cache_prompt_enabled("anthropic/claude-haiku-5.5", None)
    assert not cache_prompt_enabled("vendor/model-a", None)
    assert not cache_prompt_enabled("anthropic/claude-haiku-5.5", False)
    assert cache_prompt_enabled("vendor/model-a", True)
    cfg = parse_config({"seats": [{"type": "llm", "model": "anthropic/x"},
                                  {"type": "llm", "model": "other/y"},
                                  {"type": "llm", "model": "other/z", "cache_prompt": True}]})
    agents = build_agents(cfg, seed=1, client=object())  # type: ignore[arg-type]
    assert [a.cache_system for a in agents] == [True, False, True]
    assert "cache_prompt" not in agents[0].params  # not a request parameter


@pytest.mark.parametrize("cache", [False, True])
def test_system_prompt_cache_breakpoint(cache):
    from handel.llm.client import message_text
    from handel.runner import play_game

    seen = []

    class Spy:
        def complete(self, model, messages, **params):
            seen.append(messages)
            return OptionPickerClient(seed=1).complete(model, messages, **params)

    agent = LLMAgent(Spy(), "anthropic/x", cache_system=cache)
    from handel.bots import make_bot
    play_game([agent, make_bot("always_buy", 1)], seed=3, config=GameConfig(max_rounds=3))
    assert seen
    sysmsg = seen[0][0]
    assert sysmsg["role"] == "system"
    if cache:
        part = sysmsg["content"][0]
        assert part["cache_control"] == {"type": "ephemeral"} and part["text"].startswith("You are playing")
    else:
        assert isinstance(sysmsg["content"], str)
    assert message_text(sysmsg).startswith("You are playing HANDEL")


def test_progress_callback_and_output(tmp_path, capsys):
    from handel import play
    from handel.runner import play_game
    from handel.bots import make_bot

    rounds = []
    play_game([make_bot("random", 1), make_bot("random", 2)], seed=5,
              config=GameConfig(max_rounds=12), on_round=lambda s: rounds.append(s.round))
    assert rounds == sorted(set(rounds)) and len(rounds) >= 5

    play.main([str(ROOT / "configs/offline_demo.toml"), "--out", str(tmp_path), "--seed", "2",
               "--progress", "2"])
    err = capsys.readouterr().err
    assert "seed 2: playing" in err and "round   2 |" in err and "so far" in err

    play.main([str(ROOT / "configs/offline_demo.toml"), "--out", str(tmp_path), "--seed", "2",
               "--progress", "0"])
    assert "round" not in capsys.readouterr().err
