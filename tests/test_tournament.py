import json
import threading
from pathlib import Path

import pytest

from handel.llm import OptionPickerClient
from handel.llm.cache import (
    Budget,
    BudgetExceeded,
    CachingClient,
    ResponseStore,
    Stopped,
    prompt_key,
)
from handel.tournament import (
    JsonlFile,
    Tournament,
    format_standings,
    main,
    parse_tournament,
    schedule,
    standings,
)

ROOT = Path(__file__).resolve().parents[1]


class HashPicker:
    """A deterministic paid "API": the reply depends only on the request.

    ``crash_after`` raises a non-API error on that call, like a process dying
    mid-game.
    """

    def __init__(self, cost=0.01, crash_after=None):
        self.cost, self.crash_after, self.calls = cost, crash_after, 0
        self._lock = threading.Lock()

    def complete(self, model, messages, **params):
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.crash_after is not None and n > self.crash_after:
            raise RuntimeError("simulated crash")
        seed = int(prompt_key(model, messages, params)[:12], 16)
        c = OptionPickerClient(seed).complete(model, messages, **params)
        c.cost_usd = self.cost
        return c


def raw_config(seeds=2, rotate=True, budget=None, parallel=1, paid=False, rounds=40):
    llm = ({"type": "llm", "model": "test/a", "name": "alpha"}, {"type": "llm", "model": "test/b", "name": "beta"}) \
        if paid else ({"type": "llm", "model": "fake/random"}, {"type": "llm", "model": "fake/sloppy"})
    t = {"seeds": seeds, "rotate": rotate, "parallel": parallel}
    if budget is not None:
        t["budget_eur"] = budget
    return {"tournament": t, "game": {"max_rounds": rounds}, "llm": {"usd_to_eur": 1.0},
            "seats": [*llm, {"type": "bot", "bot": "heuristic"}, {"type": "bot", "bot": "always_buy"}]}


def quiet(_msg):
    pass


# --- config and schedule -------------------------------------------------------------


def test_schedule_rotates_every_entrant_through_every_seat():
    spec = parse_tournament(raw_config(seeds=3))
    games = schedule(spec)
    assert len(games) == 12 and games[0].game_id == "seed1_rot0" and games[-1].seed == 3
    for sd in spec.seeds:
        lineups = [g.lineup for g in games if g.seed == sd]
        for seat in range(4):
            assert sorted(lu[seat] for lu in lineups) == [0, 1, 2, 3]
    assert len(schedule(parse_tournament(raw_config(seeds=3, rotate=False)))) == 3
    assert len(schedule(parse_tournament(raw_config(seeds=3, rotate=2)))) == 6


@pytest.mark.parametrize("change, fragment", [
    ({"rounds": 3}, "unknown [tournament]"),
    ({"rotate": 9}, "rotate must be"),
    ({"seeds": 0}, "at least 1"),
])
def test_tournament_config_errors(change, fragment):
    raw = raw_config()
    raw["tournament"].update(change)
    with pytest.raises(ValueError, match=fragment.replace("[", r"\[")):
        parse_tournament(raw)


def test_entrant_names_must_be_unique():
    raw = raw_config()
    raw["seats"][3] = {"type": "bot", "bot": "heuristic"}
    with pytest.raises(ValueError, match="unique"):
        parse_tournament(raw)


def test_jsonl_drops_a_torn_last_line(tmp_path):
    f = JsonlFile(tmp_path / "x.jsonl")
    f.append({"a": 1})
    f.append({"a": 2})
    with open(f.path, "a") as fh:
        fh.write('{"a": 3, "b"')  # crash mid-write
    assert f.read() == [{"a": 1}, {"a": 2}]
    f.append({"a": 4})
    assert f.read() == [{"a": 1}, {"a": 2}, {"a": 4}]


# --- cache and budget -------------------------------------------------------------------


def test_prompt_key_is_stable_and_specific():
    m = [{"role": "user", "content": "hi"}]
    assert prompt_key("x", m, {"a": 1, "b": 2}) == prompt_key("x", m, {"b": 2, "a": 1})
    assert prompt_key("x", m, {"a": 1}) != prompt_key("y", m, {"a": 1})
    assert prompt_key("x", m, {"a": 1}) != prompt_key("x", m, {"a": 2})


def test_caching_client_replays_and_records_spend(tmp_path):
    store = ResponseStore(tmp_path / "c.sqlite")
    api = HashPicker(cost=0.5)
    c = CachingClient(api, store, Budget(store, None, 1.0))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "DECISION: buy\nOPTIONS: []"}]
    a = c.complete("m", msgs, max_tokens=5)
    b = c.complete("m", msgs, max_tokens=5)
    assert api.calls == 1 and (c.hits, c.misses) == (1, 1)
    assert b.text == a.text and b.cache_hit and not a.cache_hit and b.cost_usd == 0.5
    c.complete("m", msgs, max_tokens=6)
    assert api.calls == 2 and store.spent_usd() == pytest.approx(1.0) and len(store) == 2
    store.close()
    again = ResponseStore(tmp_path / "c.sqlite")  # persisted
    assert again.spent_usd() == pytest.approx(1.0) and again.get(prompt_key("m", msgs, {"max_tokens": 5}))


def test_budget_blocks_paid_calls_but_not_cached_ones(tmp_path):
    store = ResponseStore(tmp_path / "c.sqlite")
    budget = Budget(store, limit_eur=1.0, usd_to_eur=1.0)
    c = CachingClient(HashPicker(cost=1.0), store, budget)
    msgs = [{"role": "user", "content": "x"}]
    c.complete("m", msgs)
    assert c.complete("m", msgs).cache_hit  # already paid for: still served
    with pytest.raises(BudgetExceeded):
        c.complete("m", [{"role": "user", "content": "y"}])
    budget.stop()
    with pytest.raises(Stopped):
        c.complete("m", msgs)


# --- whole tournaments -----------------------------------------------------------------


def test_offline_tournament_runs_and_resumes(tmp_path):
    spec = parse_tournament(raw_config(seeds=2, parallel=2))
    out = tmp_path / "t"
    first = Tournament(spec, out, echo=quiet).run(limit_games=3)
    assert first.played == 3 and first.already_done == 0
    rest = Tournament(spec, out, echo=quiet).run()
    assert rest.already_done == 3 and rest.played == 5 and len(rest.lines) == 8
    lines = JsonlFile(out / "games.jsonl").read()
    assert sorted(x["game_id"] for x in lines) == sorted(g.game_id for g in schedule(spec))
    rec = json.loads((out / lines[0]["record"]).read_text())
    assert rec["result"]["winner"] == lines[0]["winner_seat"] and rec["log"]
    assert lines[0]["lineup"][lines[0]["winner_seat"]] == lines[0]["winner"]
    assert Tournament(spec, out, echo=quiet).run().played == 0  # nothing left
    rows = standings(lines, spec.entrants)
    assert sum(r["wins"] for r in rows) == 8 and all(r["games"] == 8 for r in rows)
    assert "win rate" in format_standings(lines, spec.entrants, 1.0)


def test_more_seeds_extend_an_existing_tournament(tmp_path):
    out = tmp_path / "t"
    Tournament(parse_tournament(raw_config(seeds=1)), out, echo=quiet).run()
    more = Tournament(parse_tournament(raw_config(seeds=2)), out, echo=quiet).run()
    assert more.already_done == 4 and more.played == 4


def test_changed_game_config_is_refused_unless_forced(tmp_path):
    out = tmp_path / "t"
    Tournament(parse_tournament(raw_config(seeds=1, rotate=False)), out, echo=quiet).run()
    changed = parse_tournament(raw_config(seeds=1, rotate=False, rounds=41))
    with pytest.raises(SystemExit, match="different"):
        Tournament(changed, out, echo=quiet).run()
    assert Tournament(changed, out, echo=quiet, force=True).run().played == 0


def _paid(tmp_path, name, api, **kw):
    spec = parse_tournament(raw_config(seeds=1, rotate=False, paid=True, **kw))
    return Tournament(spec, tmp_path / name, client_factory=lambda: api, echo=quiet)


def test_a_crashed_game_resumes_from_the_cache_for_free(tmp_path):
    ref_api = HashPicker()
    ref = _paid(tmp_path, "ref", ref_api).run()
    total_calls = ref_api.calls
    assert ref.played == 1 and total_calls > 20

    crashing = HashPicker(crash_after=15)
    t = _paid(tmp_path, "crash", crashing)
    first = t.run()
    assert first.played == 0 and first.failed == 1
    assert JsonlFile(tmp_path / "crash" / "errors.jsonl").read()[0]["game_id"] == "seed1_rot0"

    fresh_api = HashPicker()
    second = _paid(tmp_path, "crash", fresh_api).run()
    assert second.played == 1
    line, ref_line = second.lines[0], ref.lines[0]
    assert fresh_api.calls == total_calls - 15 and line["cache_hits"] == 15
    # The resumed game is the same game, and its nominal cost is the full cost.
    for k in ("winner", "rounds", "net_worth", "bankrupt_order", "cost_usd"):
        assert line[k] == ref_line[k]
    assert line["fresh_cost_usd"] == pytest.approx(ref_line["cost_usd"] - 15 * 0.01)
    store = ResponseStore(tmp_path / "crash" / "cache.sqlite")
    assert store.spent_usd() == pytest.approx(total_calls * 0.01)  # nothing paid twice


def test_budget_stops_mid_game_and_a_bigger_budget_resumes(tmp_path):
    api = HashPicker(cost=0.01)
    stopped = _paid(tmp_path, "b", api, budget=0.10).run()
    assert stopped.played == 0 and "budget" in stopped.stop_reason
    assert api.calls == 10  # the 11th paid call was refused
    resumed = _paid(tmp_path, "b", HashPicker(cost=0.01), budget=100.0).run()
    assert resumed.played == 1 and resumed.lines[0]["cache_hits"] == 10


def test_budget_gate_does_not_start_a_game_it_cannot_pay_for(tmp_path):
    api = HashPicker(cost=0.01)
    spec = parse_tournament(raw_config(seeds=2, rotate=False, paid=True))
    first = Tournament(spec, tmp_path / "g", client_factory=lambda: api, echo=quiet).run(limit_games=1)
    per_game = first.lines[0]["fresh_cost_usd"]
    spec.budget_eur = per_game * 1.5  # half a game left
    out = Tournament(spec, tmp_path / "g", client_factory=lambda: HashPicker(), echo=quiet).run()
    assert out.played == 0 and "not enough" in out.stop_reason


def test_cli_runs_and_summarises(tmp_path, capsys):
    out = tmp_path / "cli"
    main([str(ROOT / "configs/tournament_offline.toml"), "--out", str(out), "--seeds", "1",
          "--parallel", "2"])
    text = capsys.readouterr().out
    assert "4/4 games finished" in text and "heuristic" in text
    main([str(ROOT / "configs/tournament_offline.toml"), "--out", str(out), "--seeds", "1", "--summary"])
    assert "4 games" in capsys.readouterr().out


class FlakyApi(HashPicker):
    """Rate-limited from call ``down_from`` on (after the client's own retries)."""

    def __init__(self, down_from, **kw):
        super().__init__(**kw)
        self.down_from = down_from

    def complete(self, model, messages, **params):
        from handel.llm import LLMError

        if self.calls + 1 >= self.down_from:
            self.calls += 1
            raise LLMError("gave up after 9 attempts; last error: HTTP 429")
        return super().complete(model, messages, **params)


def test_api_outage_aborts_the_game_instead_of_playing_fallbacks(tmp_path):
    t = _paid(tmp_path, "flaky", FlakyApi(down_from=12))
    first = t.run()
    assert first.played == 0 and first.failed == 1
    err = JsonlFile(tmp_path / "flaky" / "errors.jsonl").read()[0]
    assert "LLMUnavailable" in err["error"] and "429" in err["error"]
    healthy = HashPicker()
    second = _paid(tmp_path, "flaky", healthy).run()
    assert second.played == 1 and second.lines[0]["cache_hits"] == 11
    assert all(not (st or {}).get("api_errors") for st in second.lines[0]["agent_stats"])


def test_redo_api_errors_replays_tainted_games(tmp_path):
    from handel.tournament import api_error_count

    t = _paid(tmp_path, "redo", HashPicker())
    line = t.run().lines[0]
    tainted = dict(line, agent_stats=[dict(st, api_errors=2) if st else st for st in line["agent_stats"]])
    (tmp_path / "redo" / "games.jsonl").write_text(json.dumps(tainted) + "\n")
    assert api_error_count(tainted) == 4
    assert "WARNING" in format_standings([tainted], parse_tournament(raw_config(paid=True)).entrants, 1.0)
    again = _paid(tmp_path, "redo", HashPicker()).run(redo_api_errors=True)
    assert again.played == 1 and api_error_count(again.lines[0]) == 0
    assert again.lines[0]["cache_hits"] > 0  # the whole game replays from the cache
    assert len(JsonlFile(tmp_path / "redo" / "games.superseded.jsonl").read()) == 1
    assert len(JsonlFile(tmp_path / "redo" / "games.jsonl").read()) == 1
