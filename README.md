# Handel

A benchmark where LLM agents play an original property-trading board game against
each other and against rule-based bots, with a focus on **how well they negotiate trades**.

*Handel* means "trade" in both Polish and Dutch. The board is an original 40-space
circuit of 11 Polish and 11 Dutch cities; prices, rents and cards are original and
generated from simple formulas (see `src/handel/engine/board.py`). Currency: florins (fl).

> Status: **Phase 1 (engine and bots)** and **Phase 2 (LLM agent interface)** are complete.
> Multi-round negotiation, the fair-value estimator, tournaments, analysis and the
> replay viewer come next.

## Setup

```bash
uv sync
uv run pytest                    # ~200 tests, ~30 s (includes the 10k-game benchmark)
uv run pytest -m "not slow"      # skip the benchmark
uv run handel-sim --games 10000  # parallel bot simulation with a summary
uv run handel-play configs/offline_demo.toml --games 3   # LLM pipeline, offline fake models
export OPENROUTER_API_KEY=...    # then edit model ids in configs/openrouter_example.toml
uv run handel-play configs/openrouter_example.toml --seed 1
```

## LLM agents (Phase 2)

`LLMAgent` (`src/handel/llm/agent.py`) asks the model only at real decision points:

| Decision | When | Reply |
|---|---|---|
| `buy` | landed on an affordable unowned property | `{"action": "buy" \| "decline"}` |
| `jail` | turn starts in jail with more than one option | `{"action": "pay_fine" \| "use_card" \| "roll"}` |
| `manage` | after moving, if there is something to build or unmortgage, or a color group is split between you and opponents | `{"actions": [{"type": "build", "property": "Emmen"}, ...], "trade": null \| {...}}` |
| `raise_funds` | you owe more cash than you have | `{"actions": [{"type": "mortgage", "property": "..."}, ...]}` |
| `respond` | someone offers you a trade | `{"action": "accept" \| "reject", "message": "..."}` |

* **Observation:** a static system prompt (rules + full board reference, identical for all
  seats so providers can cache it) plus a per-decision prompt: players, ownership and current
  rents by group, your own recent history, recent trades, the question, a JSON `OPTIONS:`
  line listing the legal moves, and the reply format. Properties can be named loosely
  ("Lodz", "#21", "krakow glowny").
* **Validation:** multi-action replies are replayed on a cloned state, so even-building or
  cash problems in a sequence are caught with a precise message
  (`action 2 (Build Emmen (#5)): build evenly: ...`).
* **Retry and fallback:** an invalid reply gets one retry that includes the error. If the retry
  also fails, or the API errors out, a safe default is used (decline, roll, end turn, reject,
  or a least-damaging liquidation) and the decision is marked `fallback`.
* **Metrics per seat:** decisions, LLM calls, `illegal_move_rate` (decisions whose first reply
  was malformed or illegal), fallbacks, API errors, prompt/completion/cached/reasoning tokens,
  cost in USD (from OpenRouter's `usage.cost`, or estimated from `[pricing]`) and latency.
  `handel-play` writes one JSON per game with the result, stats, every prompt/reply transcript
  and the full event log, and reports cost in USD and EUR.
* **Config:** a TOML file picks a model or bot per seat, with shared `[llm]` defaults and
  per-seat overrides (see `configs/`). Models named `fake/random` and `fake/sloppy` run
  offline for testing.
* **Cost estimate:** about 2,000 input tokens per call (about 1,200 of them the cacheable system
  prompt) and roughly one decision per round, so a full game is on the order of 150–250k
  input tokens per LLM seat.

## Engine design

```
src/handel/
  engine/
    board.py    static board: spaces, groups, prices, rent ladders
    cards.py    Kans ("chance") and Skarbiec ("treasury") decks, as data
    actions.py  immutable action types + TradeOffer
    state.py    GameState / PlayerState / GameConfig, Phase and Stage enums
    rules.py    new_game, legal_actions, validate, apply (all rule logic)
  bots/         RandomBot, AlwaysBuyBot, HeuristicBot, shared default policies
  llm/          observation + prompts, reply parsing, OpenRouter and fake clients, LLMAgent
  config.py     TOML seat config -> agents
  runner.py     play_game (agent loop), simulate_many (multiprocessing)
  play.py       handel-play CLI: play configured games, save records
```

* **State machine.** `apply(state, action)` performs one decision, then runs the
  automatic flow (dice, movement, rent, cards, debt settlement, turn order) until the
  next real decision. `state.phase` says what kind of decision is pending
  (`JAIL`, `BUY`, `MANAGE`, `RAISE_FUNDS`, `TRADE_RESPONSE`) and
  `state.decision_player` says who must make it.
* **Deterministic.** The dice RNG lives in the state, so a seed fully determines a
  game for deterministic agents. `state.clone()` snapshots everything, including
  the RNG (this is what Monte Carlo rollouts will use in Phase 4).
* **Single source of truth for legality.** `validate(state, player, action)`
  returns `None` or a human-readable reason (later fed back to LLMs on illegal moves).
  `legal_actions` enumerates exactly the actions that validate; a property test
  checks the two agree on every position at many random game states.
* **In-place mutation for speed.** `apply` mutates; `step` is a copy-on-write wrapper.
* **Parametric trades.** `ProposeTrade(TradeOffer(...))` can't be enumerated; it is
  legal in the `MANAGE` phase whenever `validate` accepts the specific offer.
  Offers carry a free-text `message` for the negotiation phase.

## Rules implemented

Buying; rent (doubled on unimproved complete sets); stations (rent by count) and
utilities (dice multiplier); houses and hotels with even building and selling and a
finite supply (32 houses, 12 hotels); mortgages (50% value, 10% interest to lift,
10% interest when a mortgaged property changes hands); jail (Go-to-jail space,
cards, three doubles; pay the fine, use a card, or roll for doubles, with a forced
fine after the third failed roll); 32 cards; income and tourist taxes; debts with a
raise-funds phase (sell buildings, mortgage); bankruptcy to a player (they get
everything) or to the bank (properties return unowned); trades of properties and
cash; a turn cap (default 150 rounds) after which the highest net worth wins.

### Deliberate simplifications

* **No auctions yet.** A declined or unaffordable property stays with the bank.
  `GameConfig.auctions` and a reserved `Phase` slot keep room for them.
* Building, mortgaging and trading happen once per turn, after moving (`MANAGE`),
  not during the buy or jail decision.
* Cash in a trade flows in one direction only; at most one proposal per turn.
* If a player goes bankrupt while owing several creditors (e.g. "pay every
  player"), the first unpaid creditor receives the assets.
* Breaking a hotel during a housing shortage drops the street to as many houses
  as the bank has, which can leave a group uneven.
* Net worth (for the turn cap) = cash + property price (minus the payoff if
  mortgaged) + buildings at cost.

## Bots

| Bot | Behaviour |
|---|---|
| `random` | Uniform over legal actions; ends turn with p = 0.5 when managing; never proposes trades. |
| `always_buy` | Buys whatever it can afford, builds whenever possible, never trades. |
| `heuristic` | Keeps a cash buffer that grows with opponents' rent threat; buys when affordable or when it completes or blocks a set; builds where rent gain per florin is highest; leaves jail early and stays late; liquidates the least valuable assets first; proposes set-completing swaps or cash purchases, judged by a valuation model that rewards completed sets and penalises handing an opponent one. |
| `heuristic_notrade` | Same, but never proposes trades (it still answers them). |

## Phase 1 findings (bot vs bot)

| Table (10,000 games, 8 workers) | Time | Avg rounds | Ends in bankruptcy |
|---|---|---|---|
| heuristic ×2, always_buy, random | 18 s | 87 | 84% |
| heuristic ×4 | 15 s | 71 | 97% |
| heuristic_notrade ×4 | 17 s | 128 | 29% |
| always_buy ×4 | 8 s | 129 | 26% |

* **Trading is what makes games decisive.** Without trades, complete sets only form
  through dice luck and most games hit the turn cap.
* **Heads-up against a partner who never trades, careful play gains little**:
  heuristic vs always_buy is about 48/52. Proposing trades is what wins: at a table
  of two trading and two non-trading heuristics, the trading seats win ~72%.
  This supports the benchmark's premise that negotiation is the key skill.
* **Seat 0 has an edge** (~29% vs 25% expected at a 4-heuristic table). Tournaments
  will rotate seats so this cancels out.
