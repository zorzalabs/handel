/* Handel replay viewer: plays back a replay exported by `handel-export`.
 * No build step and no dependencies; open via a static server (see README). */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const fmt = (x) => x.toLocaleString("en-US");
  const KEY_EVENTS = new Set(["buy", "build", "trade_proposed", "trade_countered", "trade_accepted",
    "trade_rejected", "bankrupt", "game_over", "jailed", "mortgage"]);

  const S = { doc: null, i: 0, timer: null, cells: [], firstFrameOfRound: [] };

  // --- loading ------------------------------------------------------------------------

  async function init() {
    try {
      const saved = localStorage.getItem("handel-theme");
      if (saved) document.documentElement.dataset.theme = saved;
    } catch (_) { /* storage may be blocked */ }
    bindControls();
    const params = new URLSearchParams(location.search);
    let list = [];
    try {
      const r = await fetch("replays/index.json", { cache: "no-store" });
      if (r.ok) list = await r.json();
    } catch (_) { /* opened from file:// or no index: the file picker still works */ }
    const sel = $("replay-select");
    for (const item of list) {
      const o = el("option", "", item.title || item.file);
      o.value = "replays/" + item.file;
      sel.append(o);
    }
    sel.hidden = list.length === 0;
    const wanted = params.get("replay") || (list[0] && "replays/" + list[0].file);
    if (wanted) {
      sel.value = wanted;
      await loadUrl(wanted);
    } else {
      $("meta-line").textContent = "Open a replay file exported with handel-export.";
    }
  }

  async function loadUrl(url) {
    $("meta-line").textContent = "Loading " + url + "…";
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      load(await r.json());
      const u = new URL(location.href);
      u.searchParams.set("replay", url);
      history.replaceState(null, "", u);
    } catch (e) {
      $("meta-line").textContent = "Could not load " + url + " (" + e.message + "). Serve the viewer folder " +
        "with a local web server, or use Open file.";
    }
  }

  function load(doc) {
    if (!doc || doc.format !== "handel-replay/1") {
      $("meta-line").textContent = "Not a Handel replay file.";
      return;
    }
    pause();
    S.doc = doc;
    S.firstFrameOfRound = [];
    doc.frames.forEach((f, k) => { if (S.firstFrameOfRound[f.round] === undefined) S.firstFrameOfRound[f.round] = k; });
    const m = doc.meta;
    const who = m.players.map((p, k) => `P${k} ${p.name}`).join(" · ");
    $("meta-line").textContent = `${m.game_id} · seed ${m.seed} · ${who}`;
    buildBoard();
    buildPlayers();
    buildChart();
    buildTicks();
    const notes = [];
    if (m.valued) notes.push("Win probabilities and offer values are Monte Carlo estimates: every seat is " +
      "replayed by the heuristic bot from that point (Phase 4), so they show value if the rest of the game " +
      "were played like the bot.");
    if (m.players.some((p) => p.kind === "rl")) notes.push("For the RL agent the Reasoning tab shows its three " +
      "most likely options with their probabilities and its value estimate. The win chance assumes a table of " +
      "players as strong as itself, so it reads low against weaker players.");
    if (m.redactions) notes.push(`${m.redactions} uses of a brand-like word in model-written text are shown as "full set".`);
    $("footnote").textContent = notes.join(" ");
    $("scrubber").max = String(doc.frames.length - 1);
    setFrame(0);
  }

  // --- board --------------------------------------------------------------------------

  function cellPlace(i) {
    if (i <= 10) return { row: 1, col: i + 1, side: "top" };
    if (i <= 20) return { row: i - 9, col: 11, side: "right" };
    if (i <= 30) return { row: 11, col: 31 - i, side: "bottom" };
    return { row: 41 - i, col: 1, side: "left" };
  }

  function buildBoard() {
    const board = $("board");
    board.querySelectorAll(".cell").forEach((c) => c.remove());
    S.cells = S.doc.board.map((sp, i) => {
      const { row, col, side } = cellPlace(i);
      const corner = i % 10 === 0;
      const c = el("div", `cell side-${side}${corner ? " corner" : ""}${sp.group < 0 && !corner ? " special" : ""}`);
      c.style.gridRow = String(row);
      c.style.gridColumn = String(col);
      if (sp.group >= 0) {
        c.style.setProperty("--band", S.doc.groups[sp.group].color);
        c.append(el("div", "band"));
      }
      c.append(el("div", "name", sp.name));
      if (sp.price) c.append(el("div", "price", sp.price + " fl"));
      const houses = el("div", "houses");
      const tokens = el("div", "tokens");
      c.append(houses, tokens);
      const groupName = sp.group >= 0 ? ` (${S.doc.groups[sp.group].name})` : "";
      c.title = sp.name + groupName + (sp.price ? `, ${sp.price} fl` : "");
      board.append(c);
      return { c, houses, tokens, owner: null };
    });
  }

  function renderBoard(f, prev) {
    S.doc.board.forEach((sp, i) => {
      const cell = S.cells[i];
      const o = f.owner[i];
      cell.c.classList.toggle("mortgaged", !!f.mortgaged[i]);
      if (cell.owner) { cell.owner.remove(); cell.owner = null; }
      if (o >= 0) {
        const b = el("span", "owner", "P" + o);
        b.dataset.p = String(o);
        b.title = `Owned by ${S.doc.meta.players[o].name}` + (f.mortgaged[i] ? " (mortgaged)" : "");
        cell.c.append(b);
        cell.owner = b;
      }
      cell.houses.replaceChildren();
      const h = f.houses[i];
      if (h === 5) cell.houses.append(el("span", "hotel", "H"));
      else for (let k = 0; k < h; k++) cell.houses.append(el("span", "house"));
      cell.tokens.replaceChildren();
      if (prev && (prev.owner[i] !== o || prev.houses[i] !== h)) {
        cell.c.classList.remove("flash");
        void cell.c.offsetWidth;
        cell.c.classList.add("flash");
      }
    });
    f.pos.forEach((pos, p) => {
      if (f.out[p]) return;
      const t = el("span", "token" + (f.jail[p] ? " jailed" : ""), "P" + p);
      t.dataset.p = String(p);
      t.title = S.doc.meta.players[p].name + (f.jail[p] ? " (in jail)" : "");
      S.cells[pos].tokens.append(t);
    });
  }

  function renderCenter(f, i) {
    const last = i === S.doc.frames.length - 1;
    $("round-label").textContent = last ? `Game over · round ${f.round + 1}` : `Round ${f.round + 1}`;
    const dice = $("dice");
    dice.replaceChildren();
    if (f.dice[0]) f.dice.forEach((d) => dice.append(el("span", "die", String(d))));
    const list = $("narration");
    list.replaceChildren();
    // a few earlier lines, faded, give context for what happens now
    const past = [];
    for (let k = i - 1; k >= 0 && past.length < 6; k--) {
      const evs = S.doc.frames[k].events;
      for (let j = evs.length - 1; j >= 0 && past.length < 6; j--) past.unshift(evs[j]);
    }
    const line = (ev, cls) => {
      const li = el("li", cls);
      if (ev.p >= 0) li.append(badge(ev.p));
      li.append(el("span", "", ev.t));
      list.append(li);
    };
    for (const ev of past) line(ev, "past");
    if (past.length) list.append(el("li", "sep"));
    const lines = f.events.length ? f.events : [{ k: "", p: -1, t: "…" }];
    for (const ev of lines) line(ev, KEY_EVENTS.has(ev.k) ? "key" : "");
    const box = $("on-table");
    const neg = f.trade ? S.doc.negotiations.find((n) => n.tid === f.trade) : null;
    if (!neg) { box.hidden = true; return; }
    const steps = neg.steps.filter((s) => s.frame <= i);
    const cur = steps[steps.length - 1];
    box.hidden = !cur;
    if (!cur) return;
    box.replaceChildren(el("div", "label", `Offer on the table · negotiation ${neg.tid}`), el("div", "", cur.terms));
    if (cur.message) box.append(el("div", "bubble", `“${cur.message}”`));
  }

  function badge(p) {
    const b = el("span", "pb", "P" + p);
    b.dataset.p = String(p);
    return b;
  }

  // --- players ------------------------------------------------------------------------

  function buildPlayers() {
    const root = $("players");
    root.replaceChildren();
    S.doc.meta.players.forEach((pl, p) => {
      const card = el("div", "player");
      card.id = "player-" + p;
      const row = el("div", "row");
      row.append(badge(p), el("span", "pname", pl.name));
      const kind = el("div", "kind", pl.kind === "llm" ? pl.model : pl.kind === "rl" ? `RL agent · ${pl.model}` : `bot · ${pl.model}`);
      const money = el("div", "money");
      money.append(el("b", "cash"), el("span", "worth muted small"));
      card.append(row, kind, money, el("div", "status"));
      root.append(card);
    });
  }

  function renderPlayers(f, i) {
    const last = i === S.doc.frames.length - 1;
    S.doc.meta.players.forEach((pl, p) => {
      const card = $("player-" + p);
      card.classList.toggle("turn", !last && f.current === p && !f.out[p]);
      card.classList.toggle("out", !!f.out[p]);
      card.querySelector(".cash").textContent = fmt(f.cash[p]) + " fl";
      card.querySelector(".worth").textContent = "worth " + fmt(f.worth[p]);
      const owned = f.owner.filter((o) => o === p).length;
      const st = card.querySelector(".status");
      st.classList.toggle("win", last && S.doc.meta.winner === p);
      st.textContent = last && S.doc.meta.winner === p ? "Winner"
        : f.out[p] ? "Bankrupt"
        : `${owned} propert${owned === 1 ? "y" : "ies"}` + (f.jail[p] ? " · in jail" : "");
    });
  }

  // --- chart --------------------------------------------------------------------------

  const CH = { w: 360, h: 150, l: 30, r: 34, t: 8, b: 20 };

  function buildChart() {
    const tl = S.doc.timeline || [];
    $("chart-card").hidden = tl.length < 2;
    const root = $("chart");
    root.replaceChildren();
    if (tl.length < 2) return;
    const maxR = Math.max(tl[tl.length - 1].round, S.doc.frames[S.doc.frames.length - 1].round) || 1;
    const x = (r) => CH.l + (r / maxR) * (CH.w - CH.l - CH.r);
    const y = (v) => CH.t + (1 - v) * (CH.h - CH.t - CH.b);
    S.chart = { x, y, maxR };
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${CH.w} ${CH.h}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Each player's estimated win probability by round");
    const mk = (tag, attrs, parent) => {
      const n = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      (parent || svg).append(n);
      return n;
    };
    const grid = mk("g", { class: "grid" });
    const axis = mk("g", { class: "axis" });
    for (const v of [0, 0.5, 1]) {
      mk("line", { x1: CH.l, x2: CH.w - CH.r, y1: y(v), y2: y(v) }, grid);
      const t = mk("text", { x: CH.l - 6, y: y(v) + 3, "text-anchor": "end" }, axis);
      t.textContent = Math.round(v * 100) + "%";
    }
    const step = maxR > 60 ? 20 : 10;
    for (let r = 0; r <= maxR; r += step) {
      const t = mk("text", { x: x(r), y: CH.h - 4, "text-anchor": "middle" }, axis);
      t.textContent = String(r + 1);
    }
    const n = S.doc.meta.players.length;
    for (let p = 0; p < n; p++) {
      const d = tl.map((pt, k) => `${k ? "L" : "M"}${x(pt.round).toFixed(1)},${y(pt.p[p]).toFixed(1)}`).join("");
      mk("path", { d, class: "series", stroke: `var(--p${p})` });
    }
    const ends = mk("g", { class: "end" });
    // end labels: entrants ending within 8 points share one label line
    const final = tl[tl.length - 1];
    const order = [...Array(n).keys()].sort((a, b) => final.p[b] - final.p[a]);
    const groups = [];
    for (const p of order) {
      const g = groups[groups.length - 1];
      if (g && g.v - final.p[p] < 0.08) g.ps.push(p); else groups.push({ v: final.p[p], ps: [p] });
    }
    for (const g of groups) {
      const t = mk("text", { x: x(final.round) + 5, y: y(g.v) + 3 }, ends);
      t.textContent = g.ps.map((p) => "P" + p).join(" ");
    }
    S.chart.now = mk("line", { class: "now", y1: CH.t, y2: CH.h - CH.b });
    S.chart.hover = mk("line", { class: "hover-line", y1: CH.t, y2: CH.h - CH.b, visibility: "hidden" });
    S.chart.dots = [...Array(n).keys()].map((p) => mk("circle", { r: 4, class: "dot", fill: `var(--p${p})`, visibility: "hidden" }));
    const hit = mk("rect", { x: CH.l, y: 0, width: CH.w - CH.l - CH.r, height: CH.h, fill: "transparent" });
    const nearest = (evt) => {
      const box = svg.getBoundingClientRect();
      const sx = ((evt.clientX - box.left) / box.width) * CH.w;
      let best = tl[0];
      for (const pt of tl) if (Math.abs(x(pt.round) - sx) < Math.abs(x(best.round) - sx)) best = pt;
      return best;
    };
    hit.addEventListener("mousemove", (evt) => {
      const pt = nearest(evt);
      S.chart.hover.setAttribute("x1", x(pt.round));
      S.chart.hover.setAttribute("x2", x(pt.round));
      S.chart.hover.setAttribute("visibility", "visible");
      S.chart.dots.forEach((c, p) => {
        c.setAttribute("cx", x(pt.round));
        c.setAttribute("cy", y(pt.p[p]));
        c.setAttribute("visibility", "visible");
      });
      const tt = $("tooltip");
      tt.replaceChildren(el("div", "muted small", `Round ${pt.round + 1} · click to jump`));
      [...order].sort((a, b) => pt.p[b] - pt.p[a]).forEach((p) => {
        const row = el("div", "tt-row");
        const left = el("span", "");
        left.append(badge(p), document.createTextNode(" " + S.doc.meta.players[p].name));
        row.append(left, el("b", "", Math.round(pt.p[p] * 100) + "%"));
        tt.append(row);
      });
      tt.hidden = false;
      tt.style.left = Math.min(evt.clientX + 14, innerWidth - 270) + "px";
      tt.style.top = evt.clientY + 14 + "px";
    });
    hit.addEventListener("mouseleave", () => {
      $("tooltip").hidden = true;
      S.chart.hover.setAttribute("visibility", "hidden");
      S.chart.dots.forEach((c) => c.setAttribute("visibility", "hidden"));
    });
    hit.addEventListener("click", (evt) => {
      const pt = nearest(evt);
      const k = S.firstFrameOfRound[pt.round];
      if (k !== undefined) { pause(); setFrame(k); }
    });
    root.append(svg);
  }

  function renderChart(f) {
    if (!S.chart || !S.chart.now) return;
    const cx = S.chart.x(f.round);
    S.chart.now.setAttribute("x1", cx);
    S.chart.now.setAttribute("x2", cx);
  }

  // --- side panels --------------------------------------------------------------------

  function valueLine(v, actor, other) {
    if (!v) return null;
    const line = el("div", "value");
    const part = (p) => {
      const d = v.d[p] * 100;
      const s = el("span", d > 2 * v.se[p] * 100 ? "up" : d < -2 * v.se[p] * 100 ? "down" : "");
      s.textContent = `P${p} ${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}`;
      return s;
    };
    line.append(document.createTextNode("If accepted: "), part(other), document.createTextNode(", "),
      part(actor), document.createTextNode(" pts win prob."));
    if (v.fair_extra !== null && v.fair_extra !== undefined) {
      const amt = Math.abs(v.fair_extra);
      const to = v.fair_extra > 0 ? other : actor;
      line.append(document.createTextNode(` Fair: ${amt} fl more to P${to}${v.rough ? " (rough)" : ""}.`));
    }
    return line;
  }

  function renderNegotiations(i) {
    const root = $("panel-neg");
    root.replaceChildren();
    const f = S.doc.frames[i];
    const shown = S.doc.negotiations.filter((n) => n.steps.length && n.steps[0].frame <= i).reverse();
    if (!shown.length) { root.append(el("div", "empty", "No negotiations yet.")); return; }
    for (const n of shown) {
      const box = el("div", "neg" + (f.trade === n.tid ? " live" : ""));
      const head = el("div", "neg-head");
      head.append(el("span", "", `#${n.tid} · round ${n.round + 1} ·`), badge(n.between[0]),
        el("span", "", "→"), badge(n.between[1]));
      const done = n.reply && n.reply.frame <= i;
      head.append(el("span", "outcome " + (done ? n.outcome : ""), done ? n.outcome : (f.trade === n.tid ? "open" : "")));
      box.append(head);
      for (const s of n.steps.filter((s) => s.frame <= i)) {
        const other = n.between[0] === s.p ? n.between[1] : n.between[0];
        const st = el("div", "step");
        const who = el("div", "who");
        who.append(badge(s.p), el("span", "", s.kind === "propose" ? "proposes" : "counters"));
        st.append(who, el("div", "terms", s.terms));
        if (s.message) st.append(el("div", "bubble", s.message));
        for (const pr of s.promises) st.append(el("div", "promise", pr));
        const v = valueLine(s.value, s.p, other);
        if (v) st.append(v);
        box.append(st);
      }
      if (done) {
        const st = el("div", "step");
        const who = el("div", "who");
        who.append(badge(n.reply.p), el("span", "", n.reply.kind === "accept" ? "accepts" : "rejects"));
        st.append(who);
        if (n.reply.message) st.append(el("div", "bubble", n.reply.message));
        box.append(st);
      }
      root.append(box);
    }
  }

  function renderReasoning(i) {
    const root = $("panel-think");
    root.replaceChildren();
    const notes = S.doc.reasoning.filter((r) => r.frame <= i).slice(-60).reverse();
    if (!notes.length) {
      root.append(el("div", "empty", "The LLM players' stated reasoning and the RL agent's top options appear here."));
      return;
    }
    for (const r of notes) {
      const n = el("div", r.rl ? "note rl" : "note");
      const who = el("div", "who");
      who.append(badge(r.p), el("span", "", `${S.doc.meta.players[r.p].name} · ${r.kind} · round ${S.doc.frames[r.frame].round + 1}`));
      n.append(who);
      if (r.rl) n.append(rlOptions(r.rl));
      else n.append(el("p", "", r.text));
      root.append(n);
    }
  }

  // An RL decision: its top options as probability bars, and its value estimate.
  function rlOptions(rl) {
    const box = el("div", "rl-box");
    for (const o of rl.top) {
      const row = el("div", o.action === rl.chosen ? "opt chosen" : "opt");
      const bar = el("span", "bar");
      const fill = el("i");
      fill.style.width = `${Math.max(1, Math.round(o.p * 100))}%`;
      bar.append(fill);
      row.append(el("span", "lbl", o.action), bar, el("span", "pct", `${Math.round(o.p * 100)}%`));
      box.append(row);
    }
    if (!rl.top.some((o) => o.action === rl.chosen)) box.append(el("div", "opt chosen", `took: ${rl.chosen}`));
    const v = el("div", "rl-value");
    const sign = rl.value >= 0 ? "+" : "";
    v.append(el("span", "", `value ${sign}${rl.value.toFixed(2)}`),
      el("span", "muted", ` · about ${Math.round(rl.win_chance * 100)}% to win at a table of equals`));
    box.append(v);
    return box;
  }

  function renderLog(i) {
    const root = $("panel-log");
    root.replaceChildren();
    const out = [];
    for (let k = i; k >= 0 && out.length < 200; k--) {
      const f = S.doc.frames[k];
      for (let j = f.events.length - 1; j >= 0 && out.length < 200; j--) out.push([f.round, f.events[j]]);
    }
    if (!out.length) { root.append(el("div", "empty", "Nothing has happened yet.")); return; }
    for (const [round, ev] of out) {
      const row = el("div", "ev");
      row.append(el("span", "r", "R" + (round + 1)));
      if (ev.p >= 0) row.append(badge(ev.p));
      row.append(el("span", "", ev.t));
      root.append(row);
    }
  }

  function buildTicks() {
    const root = $("ticks");
    root.replaceChildren();
    const last = S.doc.frames.length - 1 || 1;
    for (const n of S.doc.negotiations) {
      if (!n.steps.length) continue;
      const t = el("i", n.outcome === "accepted" ? "deal" : "");
      t.style.left = `calc(${(n.steps[0].frame / last) * 100}% - 1px)`;
      root.append(t);
    }
  }

  // --- playback -----------------------------------------------------------------------

  function setFrame(i) {
    if (!S.doc) return;
    const n = S.doc.frames.length;
    i = Math.max(0, Math.min(n - 1, i));
    const prev = Math.abs(i - S.i) === 1 ? S.doc.frames[S.i] : null;
    S.i = i;
    const f = S.doc.frames[i];
    renderBoard(f, prev);
    renderCenter(f, i);
    renderPlayers(f, i);
    renderChart(f);
    renderNegotiations(i);
    renderReasoning(i);
    renderLog(i);
    $("scrubber").value = String(i);
    $("frame-label").textContent = `${i + 1} / ${n}`;
    if (i === n - 1) pause();
  }

  function play() {
    if (!S.doc) return;
    if (S.i >= S.doc.frames.length - 1) setFrame(0);
    const fps = Number($("speed").value);
    clearInterval(S.timer);
    S.timer = setInterval(() => setFrame(S.i + 1), 1000 / fps);
    $("btn-play").textContent = "Pause";
  }

  function pause() {
    clearInterval(S.timer);
    S.timer = null;
    $("btn-play").textContent = "Play";
  }

  function jumpTrade(dir) {
    const starts = S.doc.negotiations.flatMap((n) => n.steps.map((s) => s.frame)).sort((a, b) => a - b);
    const target = dir > 0 ? starts.find((k) => k > S.i) : [...starts].reverse().find((k) => k < S.i);
    if (target !== undefined) { pause(); setFrame(target); selectTab("neg"); }
  }

  function selectTab(name) {
    for (const t of document.querySelectorAll(".tab")) {
      const on = t.dataset.tab === name;
      t.setAttribute("aria-selected", String(on));
      $("panel-" + t.dataset.tab).hidden = !on;
    }
  }

  function bindControls() {
    $("btn-play").addEventListener("click", () => (S.timer ? pause() : play()));
    $("btn-back").addEventListener("click", () => { pause(); setFrame(S.i - 1); });
    $("btn-fwd").addEventListener("click", () => { pause(); setFrame(S.i + 1); });
    $("btn-next-trade").addEventListener("click", () => jumpTrade(1));
    $("btn-prev-trade").addEventListener("click", () => jumpTrade(-1));
    $("scrubber").addEventListener("input", (e) => { pause(); setFrame(Number(e.target.value)); });
    $("speed").addEventListener("change", () => { if (S.timer) play(); });
    $("replay-select").addEventListener("change", (e) => loadUrl(e.target.value));
    $("replay-file").addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try { load(JSON.parse(String(reader.result))); } catch (err) { $("meta-line").textContent = "Could not read that file."; }
      };
      reader.readAsText(file);
    });
    for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => selectTab(t.dataset.tab));
    $("theme-toggle").addEventListener("click", () => {
      const root = document.documentElement;
      const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("handel-theme", root.dataset.theme); } catch (_) { /* ignore */ }
    });
    document.addEventListener("keydown", (e) => {
      if (e.target.closest("input, select, textarea")) return;
      if (e.key === " ") { e.preventDefault(); S.timer ? pause() : play(); }
      else if (e.key === "ArrowRight") { pause(); setFrame(S.i + 1); }
      else if (e.key === "ArrowLeft") { pause(); setFrame(S.i - 1); }
      else if (e.key === "]") jumpTrade(1);
      else if (e.key === "[") jumpTrade(-1);
    });
  }

  init();
})();
