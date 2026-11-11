# Walkthrough: an inspection-and-repair discussion on toy-ca

A sequence to type into the app's two panels, to simulate how a decision discussion
unfolds after an earthquake. First the decision is set: which bridge to reopen so that as
many towns as possible can reach a key destination (n1 or n2). Then observations arrive one
at a time (T1 to T4, as in the toy dispatch demo), and the decision's options are updated
after each. It takes about 5 minutes.

## Start

Run these commands from the app repository folder, in the environment the main
README creates (it installs rsr, `rsr-duco`, from PyPI).

```powershell
conda activate quakenet
python -m src serve
```

The quakenet environment matters: miniconda's base Python lacks packages rsr needs (psutil).
Without `conda activate`, run `& "C:\Users\jbyun\AppData\Local\miniconda3\envs\quakenet\python.exe" -m src serve`.

Open http://127.0.0.1:8765, choose **toy-ca**, set **samples** to 100000 and press **Run**.
The status line then reads about `100,000 samples, 100% of their weight effective, 2 s`.

The line under the panels should read `Drafting: ollama · gpt-oss:20b`. If it says no model
is ready, start Ollama and reload the page.

## How a step works

1. Type the sentence into the panel's box and press **Draft** (or Enter). It takes 10–20 s.
2. The agent replies with what it drafted, as a short summary under its reply, and says what
   it assumed. If the sentence is unclear, it asks back instead, with choices to click.
3. If the draft isn't right, say so in words, e.g. "use e7 instead of e6". The agent drafts
   again from the conversation so far.
4. Press **Add**. It adds the latest draft as shown. The map, the lists under the results and
   the **Roadmap** tab all update.

Nothing is added until you press Add, and nothing can be added without the agent: if the
model isn't running, start Ollama first.

## The sequence

| Step | Panel | Type | The agent should draft | Then |
|---|---|---|---|---|
| Decision | Decision query | I would like to priortise inspection to maximise the number of towns with access to the key destinations | it **asks** what the operator does if a bridge is inspected to be closed **click repair** Set the decision query as, for each bridge i not yet observed, VOI_i = Σ_k P(S_k = Fail \| ev) − Σ_k Σ_x P(D_i = x \| ev) · P(S_k = Fail \| do(E_i = open), ev, D_i = x). The agent adds this query to the query list | added to the roadmap and the following inferences calculate this query |
| T1 | Evidence | The station at bridge b6 reads 0.8 g | SAObs(1.0)::b6 = 0.8 g (station) | reweighted, about 10% of the weight left |
| T2 | Evidence | Police report that e9 is closed | EdgeOpen::e9 = closed (police) | sampled again (about 1% left), about 15 s |
| T3 | Evidence | n4 can't get through to the destinations, says the police | SystemState::n4 = Fail (police) | reweighted, about 26% left |
| T4 | Evidence | The bridge near n4 has collapsed | it **asks** which: b4, b6 or b7. Click **b4**: DamageState.HazusBridge::b4 = complete | reweighted, about 22% left |

## How the decision moves

VOI_i after each step: the expected number of towns that inspecting bridge i (and repairing it
if it is found damaged) gives access back to. A bridge already observed drops out: b9 after T2
(its edge e9 is known closed), b4 after T4. The values are from one run; yours will differ by
about 0.01–0.02.

| After | b2 | b3 | b4 | b6 | every other bridge |
|---|---|---|---|---|---|
| the decision query (no evidence) | **0.20** | 0.03 | 0.14 | 0.11 | ≤ 0.02 |
| T1: b6 reads 0.8 g | 0.18 | 0.03 | **0.25** | 0.23 | ≤ 0.02 |
| T2: e9 closed | 0.18 | 0.04 | **0.26** | 0.24 | ≤ 0.02 |
| T3: n4 cut off | 0.23 | 0.14 | **0.99** | 0.95 | ≤ 0.02 |
| T4: b4 complete | 0.23 | 0.04 | observed | **0.93** | ≤ 0.02 |

- **Before any evidence**, b2 comes first: inspecting it gives back 0.20 towns.
- **T1:** strong shaking at b6, next to n4, raises P(n4 cut off) from 0.14 to 0.26. b4 and b6,
  the bridges into n4, now come before b2.
- **T2:** e9 closed barely moves the priorities. It raises P(n7 cut off) from 0.003 to 0.021.
- **T3:** with n4 known to be cut off, inspecting b4 or b6 gives back about one town each;
  b2 only 0.23.
- **T4:** b4 is known to be down, so it drops out; b6 comes first, at 0.93.

## What to look at

- **Map:** towns and edges coloured by the current P(cut off) and P(closed). The bars on
  the right give the numbers.
- **Under the bars:** the decision queries being computed, and the evidence so far, each
  marked *observation*, *assumption* (a setting changed in Assumptions before Run) or *do*.
- **Roadmap tab:** one entry per addition. Each entry's bubble shows every decision query,
  the step before (grey) beside this step (blue).
- **BN tab:** click a node for its distribution as of the latest step. Try SA(1.0)::b6
  after T1: its median moves to about 0.8 g. Or DamageState.HazusBridge::b4 after T3.
- **Clique tree tab:** the messages behind each step: m_up from each clique towards the
  root (E), m_down back.

## Your own sequences

- **Decision queries** say what the decision is about: a target (expected towns cut off,
  the probability of at least *x* towns cut off, one town's probability of being cut off,
  or one edge's probability of being closed) and either *prioritise inspection* (VOI_i per
  bridge, as above) or *track* (the target itself).
- **Evidence** is a station's reading in g (SAObs), an edge reported open or closed, a
  town reported cut off (Fail) or reachable (Survival), or an inspected damage state.
  SA itself can't be observed: only through a station's reading.
- A surprising observation leaves little weight on the samples, so the session samples
  again from where they were. Expect 10–20 s on toy and longer on ema-highway. An
  unsurprising one is taken in at once.
- To change the scenario itself (magnitude, fragility sets), set it in **Assumptions**
  and press **Run** again: that starts a new session.
