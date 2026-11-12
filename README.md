# Handel

A benchmark where LLM agents play an original property-trading board game against
each other and against rule-based bots, with a focus on **how well they negotiate trades**.

*Handel* means "trade" in both Polish and Dutch. The board is an original 40-space
circuit of 11 Polish and 11 Dutch cities; prices, rents and cards are original and
generated from simple formulas (see `src/handel/engine/board.py`). Currency: florins (fl).

> Status: **Phase 1 (engine and bots)**, **Phase 2 (LLM agent interface)** and
> **Phase 3 (negotiation)** are complete. The fair-value estimator, tournaments,
> analysis and the replay viewer come next.

## Setup

```bash
uv sync
uv run pytest                    # ~230 tests, ~30 s (includes the 10k-game benchmark)
uv run pytest -m "not slow"      # skip the benchmark
uv run handel-sim --games 10000  # parallel bot simulation with a summary
uv run handel-play configs/offline_demo.toml --games 3   # LLM pipeline, offline fake models
cp .env.example .env             # then put a regular OpenRouter API key in .env
uv run handel-play configs/claude_trio.toml --seed 1     # Claude Opus/Sonnet/Haiku vs heuristic
```

`.env` is git-ignored and read automatically; a key there overrides one exported in the
shell. OpenRouter *management* keys cannot call models (every request returns 401).
`handel-play` prints a status line every 5 rounds (`--progress N`, `0` for quiet).

## LLM agents (Phase 2)

`LLMAgent` (`src/handel/llm/agent.py`) asks the model only at real decision points:

| Decision | When | Reply |
|---|---|---|
| `buy` | landed on an affordable unowned property | `{"action": "buy" \| "decline"}` |
| `jail` | turn starts in jail with more than one option | `{"action": "pay_fine" \| "use_card" \| "roll"}` |
| `manage` | after moving, if there is something to build or unmortgage, or a color group is split between you and opponents | `{"actions": [{"type": "build", "property": "Emmen"}, ...], "trade": null \| {"to", "give", "receive", "message", "promises"}}` |
| `raise_funds` | you owe more cash than you have | `{"actions": [{"type": "mortgage", "property": "..."}, ...]}` |
| `respond` | an offer is on the table for you (a proposal or a counteroffer) | `{"action": "accept" \| "reject" \| "counter", "message": "...", "counter": null \| {"give", "receive", "promises"}}` |

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
* **Prompt caching:** the system prompt (about 2,200 tokens) carries a cache breakpoint for
  `anthropic/` models (`cache_prompt`, on by default for them); the second call reads it at a
  tenth of the price. The per-seat summary shows the cached share of input tokens.
* **Cost:** about 4,000 input tokens per call, roughly one decision per round. The first live
  game (Opus 5.5, Sonnet 5.5, Haiku 5.5 and the heuristic bot, seed 1, before caching and
  before Phase 3) took 71 rounds and 10 minutes and cost $1.93: Opus $1.36, Sonnet $0.53,
  Haiku $0.04. No illegal moves and no fallbacks in 190 decisions.

## Negotiation (Phase 3)

A negotiation is a proposal followed by up to **3 counteroffers** (`GameConfig.max_counters`,
both sides together), alternating between the same two players. Whoever faces the latest
offer may accept, reject or, while counters remain, counter it; after the third counter the
only choices are accept or reject. Counters cost no turn actions, and the proposer still gets
one negotiation per turn.

* **Chat.** Every step carries a free-text `message`, and accept/reject may carry a reply
  (`AcceptTrade("Deal.")`). All messages are public: every player sees every negotiation in
  their history, the two parties step by step, everyone else as the outcome
  (`P0 ACCEPTED after 1 counteroffer: ...`, with the messages).
* **Promises.** Offers may carry up to three short `promises` ("I will not build on Polder
  for 5 rounds"). The engine never enforces them; each is logged as a `promise` event tied
  to its negotiation, so later analysis can check whether players kept their word.
* **What the LLM sees when an offer is on the table:** the whole thread so far with messages
  and promises, which offer is current, how many counters are left, and a neutral line on the
  offer's effects for both sides ("P0 would complete the Las set and could build on it";
  "you would break up your complete Kaas set"; station counts). The effects line was added
  after Haiku, in the first live game, accepted a swap that completed an opponent's set while
  calling the street it gave away "weak".
* **Records.** Each game JSON now has `negotiations` (every thread with its steps, outcome,
  closing reply and promises, rebuilt from the event log by `handel.negotiation`) and a
  `negotiation_summary` (negotiations, deals, deals after a counter, counteroffers, promises).
* **Heuristic bot.** An offer below its acceptance threshold gets a counteroffer with the same
  properties and the cash set halfway through the range both sides accept under its
  valuation model; if that range is empty there is no deal and it rejects. After a rejection
  it waits 3 rounds per consecutive rejection (up to 15) before proposing to that player again.

### Phase 3 findings (bot vs bot, 4,000 games per table)

| Table | Proposals / game (before → after) | Deals / game | Counteroffers / game | Win rates |
|---|---|---|---|---|
| heuristic vs always_buy | 26 → 6.3 | 0 | 0 | 50 / 50 |
| heuristic ×4 | 6.9 | 3.7 | 0.07 | 30 / 26 / 24 / 21 |
| heuristic, notrade, heuristic, notrade | 6.0 | 2.8 | 0.01 | trading seats 72% |

* The cooldown stops the bot spamming a player who never trades (26 offers a game before).
* Bots rarely counter each other: a bot's own proposal already splits the model surplus, so the
  partner accepts or there is no deal. Counteroffers matter against LLMs, which lowball.
  For example, 150 fl for the street that completes their set gets a 450 fl counter.
* Win rates are unchanged from Phase 1, so the earlier findings still hold.

### First live game with negotiation (Claude trio + heuristic, seed 1)

Same seats and seed as the Phase 2 game: 71 rounds, 10 minutes, **$1.24** (was $1.93; about
50% of input tokens now come from the prompt cache). 24 negotiations, 11 counteroffers,
5 deals, all 5 reached after a counteroffer; 2 promises.

* **Real haggling.** Three deals were struck by "meeting in the middle", for example Opus's
  150 fl for Radom, Haiku's counter of 230 ("Radom completes your Klei set"), Opus's 190 and
  Haiku's acceptance.
* **The effects line worked.** In Phase 2, Haiku gave away Kraków, completing Opus's Las set,
  without noticing. This time it countered "Kraków completes Las for you, so I need more" and
  got 200 fl on top of the station.
* **Blocking beats survival for Sonnet.** Opus asked Sonnet for Kielce, which completes its
  Polder set, 14 times, rising to 400 fl. Sonnet refused every offer, even with 0–20 fl in
  cash, and went bankrupt in round 34. It also noted that Opus's promise was "not binding".
  Opus escalated sensibly but never found another route.
* **Haiku made the only illegal moves:** 2 of its 83 decisions (2.4%), both times trying to
  build on Tulp without owning the whole group. Both were fixed on the retry.
* **The seed decides a lot.** The dice sequence belongs to the seed, so the game ran much as
  in Phase 2: the same bankruptcy order, and Haiku won again in round 71. Model comparisons
  need many seeds with rotated seats (Phase 5).

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
  negotiation.py  rebuild negotiation threads (chat log, promises) from an event log
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
* **Parametric trades.** `ProposeTrade(TradeOffer(...))` and `CounterTrade(TradeOffer(...))`
  can't be enumerated; they are legal (in `MANAGE` and `TRADE_RESPONSE` respectively)
  whenever `validate` accepts the specific offer. `state.thread` holds the current
  negotiation and `state.trade_id` numbers negotiations; every logged step carries the id.

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
* Cash in a trade flows in one direction only; at most one negotiation per turn, opened by the
  player whose turn it is. Promises are free text and never enforced.
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
| `heuristic` | Keeps a cash buffer that grows with opponents' rent threat; buys when affordable or when it completes or blocks a set; builds where rent gain per florin is highest; leaves jail early and stays late; liquidates the least valuable assets first; proposes set-completing swaps or cash purchases, judged by a valuation model that rewards completed sets and penalises handing an opponent one; counters offers that fall short when a mutually acceptable price exists; backs off from players who keep rejecting it. |
| `heuristic_notrade` | Same, but never proposes or counters (it still accepts or rejects offers). |

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
