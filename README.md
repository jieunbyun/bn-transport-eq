# Quakenet

## 0. What it does

Quakenet supports **post-earthquake inspection prioritisation of transport networks**.
It uses a Bayesian network to connect earthquake shaking, bridge damage, road
availability and access to key destinations, then prioritises bridge inspections
according to their expected benefit.

The browser-based GUI lets you navigate the transport network, Bayesian network
and clique tree; select a decision task; add observations; and compare results
across analysis rounds in the Roadmap and BN tabs. The main example is the
Eastern Massachusetts highway benchmark network.

## 1. Quick setup

Install **Git** and **Miniconda** (or Anaconda). On Windows, run the following
commands in Anaconda Prompt; on macOS or Linux, use a terminal with Conda available.

Download the app:

```sh
git clone https://github.com/jieunbyun/quakenet.git
cd quakenet
```

Create the Python environment and start the app:

```sh
conda env create -f environment.yml
conda activate quakenet
python -m src serve --open --example ema-highway/gmm
```

The environment installs rsr (`rsr-duco` 0.3.1 or later) from PyPI. To work on rsr alongside the app,
install your rsr checkout over it instead: `python -m pip install -e ../rsr`.

The app opens at **http://127.0.0.1:8765**. Keep the terminal open while using it.
Run a baseline, select the decision task, then add observations and run another
analysis. Use dropdown observations to try the app without configuring an LLM.

To start it again later, open a terminal in the `quakenet` folder and run:

```sh
conda activate quakenet
python -m src serve --open
```

Press **Ctrl+C** in the terminal to stop the server. If port 8765 is already in
use, add `--port 8766` to the start command and open http://127.0.0.1:8766.

## 2. Tutorial video

[**Watch the tutorial: Risk modelling of bridge networks**](https://jieunbyun.github.io/research/bridge-networks/)

Five short clips cover the system and Bayesian network, clique-tree inference,
decision tasks and evidence, reading results, and customising `network.json`.

## 3. Real earthquakes: USGS ShakeMap

`toy-ca/shakemap` and `ema-highway/shakemap` take their ground motion from a real
earthquake's USGS ShakeMap instead of a scenario and a ground-motion model: each
bridge gets the map's median SA(1.0 s) and its uncertainty, ln SA being one Gaussian
field correlated between bridges. They open on the 1989 Loma Prieta earthquake (M6.9) and
the 2020 Buzzards Bay earthquake (M3.6), the one recorded map that covers every EMA
bridge. A map that misses a bridge is refused, not read as zero shaking.

**Scenario.** Choose the example, then **Earthquakes** in the header (or **Choose
earthquake** above the map). The panel lists every USGS ShakeMap of M4 or more
since 1900 that reaches at least one of the network's bridges (132 for toy-ca, 23
for Hebei, 2 for EMA), newest first. You can also type a USGS event id. Picking an
event runs it.

A map need not reach every bridge. A bridge off the map gets the map's value at
its nearest edge, including sigma. USGS sizes each map to reach where predicted
shaking has faded to light (about MMI IV), and shaking keeps falling with
distance, so this is an upper bound, not zero. The run lists each such bridge and
its distance from the map. A map that reaches no bridge is refused.
A session records the ShakeMap version it used, so reopening it after a restart
rebuilds the same map.

**Monitoring on the page.** The Earthquakes panel asks USGS every 2 minutes for
events of the last 30 days, M4 or more, with a ShakeMap near a network. One that
appears after the page was opened (an aftershock, say) is marked NEW, announced
above the map, and counted in the tab title. While a ShakeMap session is on screen,
the page also checks whether USGS has revised that map, and offers to re-run with
the new version.

**Monitoring without the page.**

```sh
python -m src shakemap search toy-ca --min-magnitude 5     # ShakeMaps around a network
python -m src shakemap fetch nc216859                      # download one (cached in .cache/shakemaps/)
python -m src shakemap watch --once                        # one poll: run each ShakeMap assembly on new maps
python -m src shakemap watch --interval 120                # or keep polling
python -m src shakemap heartbeat                           # exit 0 if the watcher is alive and well
```

`watch` polls the USGS real-time feed. When a new ShakeMap (or a revision of one)
appears near a network, it starts that network's ShakeMap session headless,
exactly as Run does. It writes the result to
`.cache/watch/results/<event>/<event>-<version>/` (or `QUAKENET_WATCH_DIR`) and
never touches the page's sessions. Every poll writes `heartbeat.json`, so run
`heartbeat` from a second cron entry to notice a dead watcher before an earthquake
does:

```
* * * * *    cd /path/to/quakenet && $PY -m src shakemap watch --once >> watch.log 2>&1
*/10 * * * * cd /path/to/quakenet && $PY -m src shakemap heartbeat || <your alert command>
```

**Email alerts on AWS Lambda.** `src/monitor/feedwatch.py` is the feed-only
variant. It needs only the standard library, so it runs as a Lambda, without the
model. It emails once per earthquake near a network when its first ShakeMap
appears; events found in one poll share one email. Deploy or update it with
`bash scripts/deploy-feedwatch.sh you@example.com`, which needs the AWS CLI logged
in. The script creates an S3 bucket for its state, an SNS topic, the function on a
2-minute schedule, and an alarm when no healthy poll has been reported for 10
minutes. Re-run it after adding a ShakeMap assembly.
`python -m src shakemap watch --fetch-only` runs the same poll locally, with no
email.

Where maps come from: `BN_SHAKEMAP_SOURCE` = `auto` (default: a local ShakeCast
server's data, `SHAKECAST_USER_DIRECTORY`, then the USGS catalogue), `shakecast`
or `comcat`.
