// Run with: node src/app/tests/test_example_switching.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const page = fs.readFileSync(path.join(__dirname, "../quakenet.html"), "utf8");
const section = (from, to) => page.slice(page.indexOf(from), page.indexOf(to));
const code = section("async function load(id)", "// ---- controls:") +
  section("async function loadSession(", "function evidenceText(") +
  section("function showSession(s)", "// ---- panel input wiring");

class Element {
  constructor(tag = "div", root = false) {
    Object.assign(this, {tag, root, children: [], parent: null, className: "", attrs: {}, dataset: {}, value: "", style: {}});
    this.classList = {add() {}, remove() {}};
  }
  get isConnected() { return this.root || !!this.parent?.isConnected; }
  append(child) { child.parent = this; this.children.push(child); }
  clear() { this.children.forEach(c => { c.parent = null; }); this.children = []; }
  set textContent(text) { this.clear(); this.text = text; }
  get textContent() { return (this.text || "") + this.children.map(c => c.textContent).join(""); }
  set innerHTML(html) { this.clear(); this.text = html; }
  get scrollHeight() { return this.children.length * 50; }
  setAttribute(key, value) { this.attrs[key] = value; }
  removeAttribute(key) { delete this.attrs[key]; }
  closest() { return this.form; }
  querySelector() { return this.button; }
  querySelectorAll(selector = "button") {
    const tags = selector.includes("input") ? ["input", "select", "button"] : ["button"];
    return this.children.flatMap(c => [...(tags.includes(c.tag) ? [c] : []), ...c.querySelectorAll(selector)]);
  }
}

const clone = value => JSON.parse(JSON.stringify(value));
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; };
const tick = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
const toy = "toy-ca/demo", ema = "ema-highway/gmm";

function session(id, completed = true, tracked = true) {
  const name = id === toy ? "Inspection priority for towns' access" : "Inspection priority for people's access";
  const query = {id: "q1", name, target: {kind: "cut_off"}, decision: "inspect"};
  const timeline = [{kind: "start", at: "12:00:00", values: {}, elapsed_s: 0.5}];
  const bridge = id === toy ? "b6" : "bridge-e0125", reading = `SAObs(1.0)::${bridge}`;
  const variables = {[reading]: {kind: "reading", states: null}};
  for (const name of id === toy ? ["b4", "b6", "b7"] : ["bridge-e0122", "bridge-e0123", "bridge-e0125"])
    variables[`DamageState.HazusBridge::${name}`] = {kind: "damage", states: ["none", "slight", "moderate", "extensive", "complete"]};
  for (const edge of id === toy ? ["e6", "e9"] : ["e0124", "e0125"])
    variables[`EdgeOpen::${edge}`] = {kind: "edge", states: ["closed", "open"]};
  variables[`SystemState::${id === toy ? "n4" : "n62"}`] = {kind: "system", states: ["Fail", "Survival"]};
  // Every session tracks a decision task (the page forces the default one); tracked = false is a session saved without.
  if (tracked) timeline.push({kind: "query", at: "12:00:01", query, values: {q1: {}}});
  if (completed)
    timeline.push({kind: "evidence", at: "12:00:02", evidence: {variable: reading}, update: "reweighted", values: {q1: {}}});
  return {n: id === toy ? 10000 : 20000, ess: 1, unit: id === toy ? "towns" : "people",
    queries: tracked ? [query] : [], evidence: completed ? {[reading]: {value: 0.6}} : {}, variables, timeline,
    scenario: {rows: {[id]: {mw: id === toy ? 6 : 7}}}, result: {marker: id, elapsed_s: 0.5}};
}

function harness(sessions = {[toy]: session(toy), [ema]: session(ema)}, storage = new Map()) {
  const ids = ["e-talk", "e-words", "e-confirm", "agent-status", "samples", "run", "stop", "new-session", "node-pop", "msg-pop", "road",
    "decision-tasks", "task-summary", "task-status", "observation-form", "observation-chat", "observation-kind", "observation-variable",
    "observation-value", "observation-value-label", "observation-reading", "observation-reading-label", "observation-source",
    "observation-error", "observation-status"];
  const nodes = Object.fromEntries(ids.map(id => ["#" + id, new Element("div", true)]));
  nodes.main = new Element("main", true);
  const controls = [];
  for (const id of ["e"]) {
    const form = new Element("form", true);
    form.button = new Element("button", true);
    nodes[`#${id}-words`].form = form;
    controls.push(form.button, nodes[`#${id}-words`]);
  }
  for (const id of ["kind", "variable", "value", "reading", "source"])
    nodes["#observation-form"].append(nodes[`#observation-${id}`]);
  nodes["#observation-form"].append(new Element("button"));
  const modes = ["form", "chat"].map(mode => { const button = new Element("button"); button.dataset.observationMode = mode; return button; });
  const painted = {}, calls = [];
  let requestNumber = 0;
  const ctx = {console, state: {epoch: 0, loading: false, ex: null, session: null, preset: {}},
    crypto: {randomUUID: () => "test-request-" + (++requestNumber)},
    $: selector => nodes[selector], esc: s => String(s ?? ""), share: () => "100%", cliquePick: {},
    store: {get: key => storage.get(key) || null, set: (key, value) => storage.set(key, value)},
    history: {replaceState() {}}, setStatus: message => { painted.status = message; },
    agentStatus() { vm.runInContext("agentReady = true; updatePanelControls();", ctx); },
    short: name => name.split("::")[1],
    evidenceText: evidence => "Observed " + evidence.variable,
    currentScenario: () => ctx.state.preset, applyScenarioControls: () => { painted.scenario = ctx.state.preset; },
    document: {createElement: tag => new Element(tag), querySelectorAll: selector =>
      selector === "[data-observation-mode]" ? modes : controls},
    setTimeout: cb => queueMicrotask(cb),
    setInterval: () => 0, clearInterval() {},          // progress polling: no server to ask here
    getJSON: async url => {
      const parsed = new URL(url, "http://localhost"), id = parsed.searchParams.get("assembly");
      return parsed.pathname === "/api/example" ? {assembly: {id}} : {session: clone(sessions[id] || null)};
    },
    post: async (url, body) => { calls.push({url, body: clone(body)}); throw Error("Unexpected calculation request"); },
    renderRoadmap: () => { nodes["#road"].textContent = ctx.state.session?.result.marker || ""; },
    renderQuakeLine() {}, checkShakemapVersion() {}, isShakemap: () => false,
  };
  for (const name of ["Controls", "Map", "Results", "BN", "Cliques", "Schema", "Assumptions"])
    ctx[`render${name}`] = () => { painted[name] = ctx.state.ex.assembly.id; };
  vm.createContext(ctx);
  vm.runInContext(code, ctx);
  return {ctx, nodes, painted, calls, storage, eval: source => vm.runInContext(source, ctx)};
}

async function check() {
  const charts = {esc: text => String(text).replaceAll("<", "&lt;")};
  vm.runInNewContext(section("function nodeChart(d)", "async function showNodeDistribution("), charts);
  const prior = {at: "12:00:01", p: [0.75, 0.25], se: [0.01, 0.01]};
  const discrete = {kind: "discrete", states: ["closed", "open"], at: "12:00:02", p: [1, 0], se: [0, 0]};
  const paired = charts.nodeChart({...discrete, previous: prior});
  assert.match(paired, /Previous round \(12:00:01\)/);
  assert.match(paired, /Current round \(12:00:02\)/);
  assert.match(paired, /width="112.5"/);
  assert.match(paired, /width="150"/);
  assert.match(paired, /width="0"/);
  assert.match(charts.nodeChart(discrete), /No previous round recorded yet/);
  const continuous = {kind: "continuous", at: "12:00:02", ln_edges: [-2, -1, 0], density: [1, 0], median_g: 0.2};
  const overlaid = charts.nodeChart({...continuous, previous: {...continuous, at: "12:00:01", density: [0, 4], median_g: 0.6}});
  assert.equal((overlaid.match(/<path /g) || []).length, 2);
  assert.match(overlaid, /fill:var\(--idle\)/);
  assert.match(overlaid, /fill:var\(--accent\)/);
  assert.match(overlaid, /L18,84/); // Current density is scaled to the previous round's higher peak.
  assert.doesNotMatch(overlaid, /NaN|Infinity/);
  new vm.Script(page.match(/<script>([\s\S]*?)<\/script>/)[1]);
  const h = harness();
  await h.ctx.load(toy);
  assert.match(h.nodes["#decision-tasks"].textContent, /number of towns/);
  assert.doesNotMatch(h.nodes["#task-summary"].textContent, /pop_k/);
  const toyChat = h.nodes["#e-talk"].textContent;
  assert.match(toyChat, /b6/);
  h.nodes["#e-words"].value = "a toy observation";
  h.ctx.setObservationMode("chat");
  await h.ctx.load(ema);
  assert.match(h.nodes["#decision-tasks"].textContent, /number of people/);
  assert.match(h.nodes["#task-summary"].textContent, /pop_k/);
  assert.match(h.nodes["#e-talk"].textContent, /bridge-e0125/);
  assert.equal(h.nodes["#samples"].value, 20000);
  assert.equal(h.nodes["#road"].textContent, ema);
  assert.equal(h.painted.scenario.rows[ema].mw, 7);
  assert.equal(h.eval("observationUI.mode"), "form");
  await h.ctx.load(toy);
  assert.equal(h.nodes["#e-talk"].textContent, toyChat);
  assert.equal(h.nodes["#e-words"].value, "a toy observation");
  assert.equal(h.eval("observationUI.mode"), "chat");
  const reopened = harness(undefined, h.storage);
  await reopened.ctx.load(toy);
  assert.equal(reopened.nodes["#e-talk"].textContent, toyChat);

  // Opening a session computes nothing, even one without a decision task. A click on the task computes it on the
  // open session, without a language model, and a second click cannot add it twice.
  for (const id of [toy, ema]) {
    const data = {[id]: session(id, false, false)}, full = harness(data);
    full.ctx.post = async (url, body) => {
      full.calls.push({url, body: clone(body)});
      const s = data[id];
      if (url.endsWith("/query")) {
        const query = {...body.query, id: "q1"};
        s.queries.push(query);
        s.timeline.push({kind: "query", at: "12:00:01", query, values: {q1: {}}});
      } else {
        s.evidence[body.variable] = {value: body.value};
        s.timeline.push({kind: "evidence", at: "12:00:02", evidence: body, update: "reweighted", values: {q1: {}}});
      }
      return clone(s);
    };
    await full.ctx.load(id);
    await tick();
    assert.equal(full.calls.length, 0);                   // nothing computed on opening
    await full.ctx.selectDecisionTask("inspection-access");
    await full.ctx.selectDecisionTask("inspection-access");
    assert.equal(full.calls.length, 1);
    assert.equal(full.nodes["#decision-tasks"].children[0].attrs["aria-pressed"], "true");
    assert.equal(full.calls[0].url, "/api/session/query");
    assert.equal(full.calls[0].body.assembly, id);
    assert.equal(full.calls[0].body.query.target.kind, "cut_off");
    assert.equal(full.calls[0].body.query.decision, "inspect");
  }

  // Before a session, choosing a task computes nothing: Run waits for one, then samples and computes it.
  const fresh = harness({[toy]: null});
  await fresh.ctx.load(toy);
  assert.equal(fresh.nodes["#run"].disabled, true);
  await fresh.ctx.selectDecisionTask("inspection-access");
  assert.equal(fresh.calls.length, 0);
  assert.equal(fresh.nodes["#run"].disabled, false);
  assert.equal(fresh.nodes["#decision-tasks"].children[0].attrs["aria-pressed"], "true");
  fresh.ctx.post = async (url, body) => {
    fresh.calls.push({url, body: clone(body)});
    return url.endsWith("/query") ? session(toy, false) : session(toy, false, false);
  };
  await fresh.ctx.run();
  assert.deepEqual(fresh.calls.map(c => c.url), ["/api/session", "/api/session/query"]);
  assert.equal(fresh.calls[1].body.query.decision, "inspect");

  // Typed observations can target any asset and always use the actual drafting endpoint.
  const live = harness({[toy]: session(toy, false), [ema]: session(ema)});
  await live.ctx.load(toy);
  live.ctx.post = async (url, body) => {
    live.calls.push({url, body: clone(body)});
    return {draft: {variable: "EdgeOpen::e9", value: "closed", source: "police"}};
  };
  await live.ctx.draft("evidence", "Police say e9 is closed");
  assert.equal(live.calls[0].url, "/api/session/draft");
  assert.equal(live.calls[0].body.kind, "evidence");
  assert.equal(live.eval("pending.evidence.variable"), "EdgeOpen::e9");
  assert.equal(live.nodes["#e-confirm"].hidden, false);
  assert.equal(live.calls.length, 1); // Drafting has not added evidence.
  await live.ctx.load(ema);
  assert.equal(live.eval("pending.evidence"), null);
  await live.ctx.load(toy);
  assert.equal(live.eval("pending.evidence.variable"), "EdgeOpen::e9");

  // Dropdown observations bypass the LLM, validate readings, and use the same evidence endpoint.
  const fields = harness({[toy]: session(toy, false)});
  await fields.ctx.load(toy);
  fields.ctx.post = async (url, body) => {
    fields.calls.push({url, body: clone(body)});
    const s = session(toy, false);
    s.evidence[body.variable] = {value: body.value};
    s.timeline.push({kind: "evidence", at: "12:00:03", evidence: body, update: "reweighted", values: {}});
    return s;
  };
  fields.nodes["#observation-reading"].value = "-0.8";
  await fields.ctx.submitObservation();
  assert.equal(fields.calls.length, 0);
  assert.match(fields.nodes["#observation-error"].textContent, /nonnegative/);
  fields.nodes["#observation-reading"].value = "0.8";
  fields.nodes["#observation-source"].value = " station ";
  await fields.ctx.submitObservation();
  assert.equal(fields.calls[0].url, "/api/session/evidence");
  assert.equal(fields.calls[0].body.variable, "SAObs(1.0)::b6");
  assert.equal(fields.calls[0].body.value, 0.8);
  assert.equal(fields.calls[0].body.source, "station");
  assert.equal(fields.eval("pending.evidence"), null);
  assert.equal(fields.ctx.observationVariables("reading").length, 0);

  for (const delayedPath of ["/api/example", "/api/session"]) {
    const race = harness(), gate = deferred(), original = race.ctx.getJSON;
    race.ctx.getJSON = async url => {
      const parsed = new URL(url, "http://localhost");
      if (parsed.pathname === delayedPath && parsed.searchParams.get("assembly") === toy) await gate.promise;
      return original(url);
    };
    const slow = race.ctx.load(toy); await tick(); await race.ctx.load(ema);
    gate.resolve(); await slow;
    assert.equal(race.ctx.state.ex.assembly.id, ema);
    assert.equal(race.nodes["#road"].textContent, ema);
    assert.equal(race.painted.Map, ema);
    assert.equal(race.nodes.main.inert, false);
  }
  const late = harness(), gate = deferred();
  await late.ctx.load(toy);
  late.ctx.post = async (url, body) => { assert.equal(body.assembly, toy); return gate.promise; };
  const calculation = late.ctx.sessionCall("/api/session/evidence", {variable: "EdgeOpen::e9", value: "closed"}, "Computing");
  await late.ctx.load(ema); gate.resolve(session(toy));
  assert.equal(await calculation, false);
  assert.equal(late.nodes["#road"].textContent, ema);

  const playing = harness({[toy]: session(toy, false), [ema]: session(ema)}), timers = [];
  await playing.ctx.load(toy);
  playing.ctx.setTimeout = cb => timers.push(cb);
  const playback = playing.ctx.say("evidence", "agent", "An old network's reply", undefined, true);
  await playing.ctx.load(ema);
  while (timers.length) { timers.shift()(); await tick(); }
  await playback;
  assert.equal(playing.calls.length, 0);
  assert.doesNotMatch(playing.nodes["#e-talk"].textContent, /old network/);

  const empty = harness({[toy]: session(toy)});
  await empty.ctx.load(toy); await empty.ctx.load(ema);
  assert.equal(empty.ctx.state.session, null);
  assert.equal(empty.nodes["#e-talk"].textContent, "Press Run first.");
  assert.equal(empty.nodes["#road"].textContent, "");

  const reply = deferred(), model = harness();
  await model.ctx.load(toy);
  model.ctx.post = () => reply.promise;
  const drafting = model.ctx.draft("evidence", "a question for the toy network");
  await tick(); await model.ctx.load(ema);
  reply.resolve({note: "A late reply for the toy network"}); await drafting;
  assert.doesNotMatch(model.nodes["#e-talk"].textContent, /late reply/);
  const stopped = harness(), cancelled = deferred();
  await stopped.ctx.load(toy);
  const previousSession = stopped.ctx.state.session;
  stopped.ctx.post = async (url, body) => {
    stopped.calls.push({url, body});
    if (url === "/api/session/stop") { cancelled.resolve(); return {stopping: true}; }
    await cancelled.promise;
    const error = new Error("Stopped"); error.cancelled = true; throw error;
  };
  const updating = stopped.ctx.sessionCall("/api/session/query", {query: {}}, "Computing");
  assert.equal(stopped.nodes["#stop"].disabled, false);
  await stopped.ctx.stopInference();
  assert.equal(await updating, false);
  assert.equal(stopped.ctx.state.session, previousSession);
  assert.equal(stopped.calls[1].url, "/api/session/stop");
  assert.equal(stopped.calls[1].body.request_id, stopped.calls[0].body.request_id);
  assert.equal(stopped.nodes["#stop"].disabled, true);
  assert.match(stopped.painted.status, /Inference stopped/);
  return "Decision tasks, observation modes, saved memory, switching isolation, and inference Stop passed.";
}

module.exports = check;
if (require.main === module) check().then(console.log).catch(error => { console.error(error); process.exitCode = 1; });
