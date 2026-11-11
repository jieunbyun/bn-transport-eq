# 1. Schema: what can exist

The schema stores **functions, not just their metadata**. Each one is a typed
signature that says what it consumes and produces, and on which table of rows it
is replicated:

```
outputs[per-row]  =  f( inputs  ;  fixed data  ;  θ )
                       ▲            ▲              ▲
                       BN parents   row attributes  a parameter set,
                       (wired by    and whole       shared by every row
                       type + via)  tables          bound to it
```

The split into three kinds of argument is what lets a few records describe a
whole network. Three Hazus parameter sets (θ) serve all 129 EMA bridges: each
bridge's class (fixed data) picks its set, and its own lat/lon (fixed data) is where
the hazard functions evaluate its shaking.

**Two drawings of all of this**, both generated from the files below by
`python -m src diagram` (add `--open` to view the HTML one), and checked by a test
that fails if either falls behind them:

| File | Holds |
|---|---|
| `diagram.html` | the interactive drawing: open it in a browser. **Model** puts the functions and the tables they read in one picture: solid wires are random variables, dashed ones the data a function reads (attributes, whole tables, the keys that route parents), dotted ones foreign keys. **Functions** and **Plates** show each half alone |
| `diagram.md` | the same as Mermaid diagrams, which GitHub renders, plus a table of quantities |
| `quantities.json` | the types that flow between functions: a unit and range, ordered states, or record fields |
| `entities.json` | the plates: tables of rows, with typed attributes and foreign keys (`ref`) |
| `functions/<id>.json` | one typed signature each: its plate, ports, the attributes and tables it reads, a JSON Schema for its instances (`instance_schema`), and the kernel that implements it |
| `meta/*.schema.json` | the JSON Schema each file here and in `instances/` obeys |

The instances that fill this schema, parameter sets filed by the role of their
function, are in [`../instances/`](../instances/README.md); the networks and
assemblies they are shown on, in [`../examples/`](../examples/README.md).

## How a function is replicated: `per` and `via`

A function's `per` is its plate. Each input names the plate its parent lives on
(`over`) and how to reach it from one row (`via`):

| `via` | Parents of one row | Example |
|---|---|---|
| `self` | the same row | `DamageState[bridge] ← SA[bridge]` |
| `<fk>` | the row this foreign key points at | e.g. `X[bridge] ← EdgeOpen[bridge.edge]`, a bridge reading its own edge's state; none in use yet |
| `<Over>.<fk>` | every `Over` row whose key points here | `EdgeOpen[edge] ← DamageState[b] for every b with b.edge = edge` |
| `*` | every row of `Over` | a system function reading every `EdgeOpen` |

A fixed scenario is not a variable: the GMM reads the one `Event` row as a whole table
(`tables: ["Event"]`), fixed data like each bridge's lat/lon. Only an uncertain
rupture would be a node, from a scenario function on `Event` feeding every bridge's
GMM through `*`.

## Intensity measures are left open

`SA1` is no longer special. The IMs are quantities in their own right (`PGA`, `PGV`,
`SA(0.3)`, `SA(1.0)`, `SA(3.0)`, named as in OpenQuake), and a port may declare its
quantity as `"{imt}"`: the IM its parameter set names. The GMM, the ShakeMap reader
and the fragility curve are all declared that way, so one function serves every IM.

Wiring is still checked on the resolved types. Hazus's curves say `imt: "SA(1.0)"`,
so they connect only to a hazard whose set also says `SA(1.0)`; a PGA hazard feeding
them is a compile error that names each bridge and what it has instead. For a templated
port the check also demands that `imt` be a required parameter whose allowed values
are all quantity ids, so every set that validates resolves to a real type.

## Hazard: two routes to the same IM

Both produce `{imt}` on the `Bridge` plate, so either feeds any fragility function
on the same IM:

- **GMM** (`hazard.gmm`): one function for every ground-motion model. Each parameter
  set is one model: its functional `form` (a kernel in `GMM_FORMS`), the `imt` it
  predicts, its coefficients and σ, and a `region` where it applies. Campbell (2003)
  is one such set. It has no parents: its fixed data are the `Event` row (magnitude,
  epicentre, depth) and the bridge's lat/lon. Making the rupture uncertain would add a
  scenario function and a `rupture` input here.

  **Which GMM is chosen by where the network is.** A binding with
  `"parameter_set_by_location": ["model.a", "model.b"]` gives each bridge the first
  listed set whose `region` polygon contains it. EMA's bridges all fall in Campbell's
  "central and eastern North America" box. A bridge no listed region covers is an
  error that names it, never a silent fallback. A query can still force one set
  everywhere with `parameter_sets`.
- **ShakeMap** (`hazard.shakemap`): the map is the event's own data, so θ names the
  event. Each bridge reads its median from `grid.xml` and its σ from `uncertainty.xml`
  at its lat/lon.

Either way the IM is one Gaussian field over the whole `Bridge` plate (`drawKind:
Joint`): ln IM ~ N(μ, D R D) with D = diag(σ), μ and σ from the function and R the
correlation between bridges. R is chosen apart from the function's own parameters:
the function names a `correlation_model` (`schema/models/spatial-correlation.json`),
and a binding picks one of its sets with `"correlation": "jayaram-baker-2009.sa1p0"`
(Jayaram & Baker 2009 at the IM's period), or leaves it out for independent bridges.
A query swaps it with `"parameter_sets": {"spatial-correlation": "<set id>"}`. There
is no residual node: observations of ground motion enter only through `SAObs`.

## Fragility: one function, curves as instances

`fragility.bridge` is the signature only: an IM at a bridge in, a damage state out.
Each curve is a parameter set under `instances/fragility/`. It says what kind of
curve it is, which decides how it is wired, and gives the form's own parameters:

```json
{"id": "hazus61.HWB1",
 "values": {"form": "lognormal", "imt": "SA(1.0)", "damage_scale": "DamageState.HazusBridge",
            "parameters": {"medians": {"slight": 0.4, "moderate": 0.5, "extensive": 0.7, "complete": 0.9},
                           "beta": 0.6}}}
```

- `form` picks the kernel (`FRAGILITY_FORMS` in `src/builder/kernels/fragility.py`), as
  a GMM's does. `lognormal`'s parameters are a median of the IM per limit state and β, the standard
  deviation of ln IM: one shared `beta` or a `betas` per limit state; where
  per-state curves cross, P(DS ≥ k) is capped at P(DS ≥ k−1) and the result says how often.
- `imt` is the input's quantity, as above.
- `damage_scale` is the output's quantity, declared `"{damage_scale}"` the same way.
  The limit states the values are keyed by are that scale's states after the first,
  and the check holds them to it. `edge.open` reads `{damage_scale}` too, so an
  edge's closure policy wires only to curves on its own scale.

A curve of an existing form on an existing scale is an instance file and nothing else.
The schema grows only for a new form (a kernel and its `allOf` branch) or a new damage
scale (a quantity, and its branch in the two functions).

## What `python -m src check` refuses

- a port naming an unknown quantity, or a `via` that does not follow a real foreign key;
- parameter values that do not satisfy their function's `instance_schema`;
- a declared capability the kernel lacks (`OVERCLAIM`);
- rows with unknown or missing attributes, wrong types, or dangling foreign keys;
- a function reading an attribute that some bound row lacks;
- a parameter-set file outside `instances/<role>/` for its function's role;
- a `parameter_set` template that resolves to a missing set (it names the rows);
- a `parameter_set_by_location` that leaves some row uncovered (it names the rows);
- a templated port (`{imt}`, `{damage_scale}`) whose parameter is not a required enum of quantity ids;
- a node with no producer or with two, and any cycle among bindings.

Wiring is by resolved quantity, so a hazard on one IM cannot feed curves on another
silently: the check names the bridges and the IM they do have.

## Adding things

- **A bridge class**: add a set to `instances/fragility/hazus61-bridge-sa1.json`.
  Rows with that `hazus_class` pick it up through `"hazus61.{hazus_class}"`.
- **Another source's curves**: a new file in `instances/fragility/`, with each set's
  `form`, `imt`, `damage_scale` and values, and a Bridge attribute naming its class for
  the binding's `parameter_set` template. No schema change.
- **A curve modifier**, e.g. Hazus's span count: a Bridge attribute listed under
  `fragility.bridge`'s `attributes`. The check then lists every bridge that lacks it.
- **A GMM of an existing form**: only a parameter set for `hazard.gmm`, with its
  `form`, `imt`, coefficients, σ and `region`, then list it in the assembly's
  `parameter_set_by_location`. No code.
- **A GMM of a new form**: a function in `GMM_FORMS` (`src/builder/kernels/hazard.py`)
  returning ln median and σ, and a branch for its coefficients under `allOf` in
  `functions/hazard.gmm.json`. If it needs Vs30, add `vs30` to `Bridge` and list it
  under `attributes`.
- **Curves on another IM**: a parameter set for `fragility.bridge` with that `imt`.
- **A system function**: one that reads every `EdgeOpen` via `*` and the `Node`/`Edge`
  tables, and calls an ndtools evaluator (`src/builder/kernels/system.py`). Its rsr
  rules are extracted once by `python -m src build` and stored with the example.
- **A kernel** is a tbnpy probability class in `src/builder/kernels/`, written like
  tbnpy's own examples: its constructor takes `childs`, `parents` and the distribution's
  parameters by name, and it samples with `sample()` and evaluates `log_prob()`, in
  PyTorch. `Class.from_spec(childs, parents, spec)` builds it for one binding's rows
  from the generated model.json, reading their parameter sets and tables.

The records map one-to-one onto relational tables (quantity, entity, function,
port, parameter_set, binding), so they could be loaded into Postgres or Supabase.
They are kept as files because contributions and reviews are diffs.
