# Docker + Airflow Commands

Everything needed to set up, start, stop, inspect and repair the Airflow orchestration layer
for the welfare fraud-detection pipeline.

**Rules that apply to every command below**

- Run them from the **project root** (the folder that contains `docker-compose.yaml`, `src/`, `data/`, `airflow/`):
  ```bash
  cd /Users/macbookpro/Exd/fraud-detection-system/welfare-fraud-detection-pipeline
  ```
- `docker compose` (with a space) is Docker Compose v2. Older guides write `docker-compose`; same tool.
- Docker Desktop must be running (whale icon in the menu bar) before any `docker` command works.

---

## 1. How the pieces fit together

| Piece | Where it runs | Purpose |
|---|---|---|
| Airflow webserver | container | The UI at http://localhost:8080 |
| Airflow scheduler | container | Decides which task runs when; also fires the daily schedule |
| Airflow worker | container | Actually executes your tasks (your ETL and fraud-detection code) |
| Airflow triggerer | container | Support process for deferred tasks (unused by this DAG, needed by Airflow) |
| Airflow Postgres | container | Airflow's **own** metadata DB: run history, task states, users |
| Redis | container | Message queue between scheduler and worker |
| **Your project Postgres** | **your Mac (host)** | Your welfare data. Containers reach it as `host.docker.internal` |
| Your project folder | mounted into containers at `/opt/airflow/project` | Code edits on your Mac appear instantly inside the containers |

Two consequences worth remembering:

- Editing files in `src/` or `airflow/dags/` needs **no rebuild** (they are mounted).
- Changing `Dockerfile` or `requirements-airflow.txt` **does** need a rebuild (section 4).
- Wiping Airflow's containers/volumes never touches your welfare data: that lives in your host Postgres and in `data/`.

---

## 2. First-time setup (from scratch)

Do this once, in a fresh project folder.

### 2.1 Check Docker works
```bash
docker --version
docker compose version
docker info > /dev/null && echo "docker daemon OK"
```
Prints versions and confirms the Docker daemon is reachable. If the last line fails, start Docker Desktop.

### 2.2 Make the Airflow folders
```bash
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/config
```
Airflow expects these four folders. Your DAG file goes in `airflow/dags/`
(the folder must be named `dags`, not `dag`, or Airflow will never see the file).

### 2.3 Download the official compose file
```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.10.5/docker-compose.yaml'
```
`-L` follows redirects, `-f` fails loudly on HTTP errors, `-O` saves it as `docker-compose.yaml`.
Airflow 2.10.5 is pinned because the DAG was tested against it.

### 2.4 Add compose settings to `.env`
```bash
printf '\nAIRFLOW_PROJ_DIR=./airflow\nAIRFLOW_UID=50000\n' >> .env
```
- `AIRFLOW_PROJ_DIR=./airflow` tells the compose file that dags/logs/plugins/config live in `./airflow`.
- `AIRFLOW_UID=50000` is the user ID Airflow runs as inside the containers (the default on macOS).
- `>>` **appends**, so your existing database credentials in `.env` are untouched.

### 2.5 Create the custom image recipe
```bash
cat > Dockerfile <<'EOF'
FROM apache/airflow:2.10.5
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
EOF

cat > requirements-airflow.txt <<'EOF'
pandas>=2.0,<3
rapidfuzz
recordlinkage
scikit-learn
python-dotenv
matplotlib
EOF
```
The stock Airflow image lacks the libraries your fraud-detection code imports.
The Dockerfile starts from the official image and adds them.

### 2.6 Patch the compose file
```bash
python3 - <<'PYEOF'
import re, pathlib
p = pathlib.Path("docker-compose.yaml"); s = p.read_text()

def sub(old, new, label):
    global s
    assert old in s, f"could not find {label} - compose file differs from expected"
    s = s.replace(old, new, 1)

s, n = re.subn(r"(?m)^  image: \$\{AIRFLOW_IMAGE_NAME:-[^}]*\}\n",
               "  image: welfare-airflow:2.10.5\n  build: .\n", s, count=1)
assert n == 1, "could not find the image: line"
s = s.replace("  # build: .\n", "", 1)
sub("AIRFLOW__CORE__LOAD_EXAMPLES: 'true'",
    "AIRFLOW__CORE__LOAD_EXAMPLES: 'false'\n    PROJECT_ROOT: /opt/airflow/project\n"
    "    PYTHONPATH: /opt/airflow/project\n    PROJECT_DB_HOST: host.docker.internal", "LOAD_EXAMPLES")
sub("    - ${AIRFLOW_PROJ_DIR:-.}/plugins:/opt/airflow/plugins\n",
    "    - ${AIRFLOW_PROJ_DIR:-.}/plugins:/opt/airflow/plugins\n    - .:/opt/airflow/project\n", "plugins volume")
sub('  user: "${AIRFLOW_UID:-50000}:0"\n',
    '  user: "${AIRFLOW_UID:-50000}:0"\n  extra_hosts:\n    - "host.docker.internal:host-gateway"\n', "user line")
p.write_text(s); print("compose patched")
PYEOF
```
What each edit does:

| Edit | Why |
|---|---|
| `image:` line becomes `build: .` plus a fixed image tag | Build the custom image from your Dockerfile; all services share one tag |
| `LOAD_EXAMPLES: 'false'` | Hides Airflow's built-in demo DAGs |
| `PROJECT_ROOT`, `PYTHONPATH` | Tell the DAG and Python where your project is mounted |
| `PROJECT_DB_HOST: host.docker.internal` | Where your host Postgres is, seen from inside a container |
| `- .:/opt/airflow/project` | Mounts your whole project into every Airflow container |
| `extra_hosts ... host-gateway` | Makes `host.docker.internal` resolve (needed on Linux, harmless on macOS) |

> **Do not rename `PROJECT_DB_HOST` to `DB_HOST`.** Airflow's own startup script reads a variable
> called `DB_HOST` to find its Redis broker. Setting it to your Mac's address makes the scheduler
> and worker loop forever on "connection refused" and every run stays *queued*.
> The DAG copies `PROJECT_DB_HOST` into `DB_HOST` only when a task runs.

The script stops with an error message if the compose file has a different layout than expected,
instead of silently producing a broken file. It is not safe to run twice on the same file.

### 2.7 Put the DAG in place
```bash
cp ~/Downloads/welfare_fraud_pipeline.py airflow/dags/
```
Adjust the source path to wherever the file is. (If the DAG is already committed in your repo, skip this.)

### 2.8 Build, initialise, start
```bash
docker compose build
docker compose up airflow-init
docker compose up -d
```
- `build` builds the custom image (slow the first time: it downloads Airflow and installs the libraries).
- `up airflow-init` runs a one-shot container that creates Airflow's database tables and the
  login user (`airflow` / `airflow`). It exits by itself when finished. **First time only.**
- `up -d` starts all services in the background (`-d` = detached).

Open http://localhost:8080 and log in with `airflow` / `airflow`.
New DAG files are discovered every ~5 minutes; to see it immediately, run section 5's `reserialize`.

### 2.9 Smoke tests
```bash
docker compose exec airflow-scheduler python -c "import recordlinkage, rapidfuzz, sklearn; print('deps ok')"

docker compose exec -e DB_HOST=host.docker.internal -w /opt/airflow/project airflow-scheduler \
  python -c "from src.database.connection import get_connection as g; c=g(); print('db ok', c.closed==0); c.close()"
```
- First: the extra libraries are installed in the image.
- Second: code inside the container can reach your host Postgres.
  `-e` sets an environment variable for this command only, `-w` sets the working directory.
  (`-e DB_HOST=...` is needed here because only the DAG maps `PROJECT_DB_HOST` to `DB_HOST` automatically.)

Do not trigger the DAG until both print OK.

---

## 3. Daily use: start, stop, inspect

### Start
```bash
docker compose up -d
```
Starts (or re-creates, if the configuration changed) every service in the background. Safe to repeat.

### Stop
```bash
docker compose stop      # stop containers, keep them (fast restart with `docker compose start`)
docker compose down      # stop AND remove containers + network (Airflow history is kept)
```
Use `stop` for a short break, `down` for a clean shutdown. Both keep Airflow's run history,
because it lives in a Docker volume.

### Check status
```bash
docker compose ps
```
Lists each service and its health. Healthy state: every service says `Up ... (healthy)`.
`(health: starting)` for the first minute after a start is normal.

### Read logs
```bash
docker compose logs --tail=50 airflow-scheduler          # last 50 lines
docker compose logs -f --tail=50 airflow-scheduler       # keep following (Ctrl+C to stop following)
docker compose logs --tail=50 airflow-worker
```
Service names: `airflow-webserver`, `airflow-scheduler`, `airflow-worker`, `airflow-triggerer`, `postgres`, `redis`.

### Restart one service
```bash
docker compose restart airflow-scheduler
```

### Which containers are running (any project)
```bash
docker ps
```
Use this if port 8080 is already taken by another project's Airflow. Stop that project first
(run `docker compose down` in *its* folder).

---

## 4. Changing things

| You changed | Do this |
|---|---|
| A file in `src/` (ETL, fraud detection) | Nothing. It is mounted; the next task run uses it |
| `airflow/dags/welfare_fraud_pipeline.py` | Nothing. Re-read within ~30 seconds. To force: `reserialize` (section 5) |
| `requirements-airflow.txt` or `Dockerfile` | `docker compose build` then `docker compose up -d` |
| `docker-compose.yaml` or `.env` | `docker compose up -d` (recreates only what changed) |

```bash
docker compose build              # rebuild the image
docker compose build --no-cache   # rebuild from zero if a cached layer is misbehaving
docker compose up -d              # restart services on the new image
```

### Full reset (last resort)
```bash
docker compose down --volumes --remove-orphans
docker compose up airflow-init
docker compose up -d
```
`--volumes` deletes Airflow's metadata database: run history, task states, the login user.
`--remove-orphans` removes leftover containers from older compose layouts.
It does **not** touch your welfare data (host Postgres, `data/`). Because the metadata is gone,
run `airflow-init` again afterwards.

---

## 5. Airflow CLI (runs inside the scheduler container)

Every command has the form `docker compose exec airflow-scheduler airflow <command>`.

```bash
# See what Airflow knows about
docker compose exec airflow-scheduler airflow dags list
docker compose exec airflow-scheduler airflow dags list-import-errors     # "No data found" = all DAG files parse fine
docker compose exec airflow-scheduler airflow dags reserialize            # force a re-scan of the dags folder now

# Run control
docker compose exec airflow-scheduler airflow dags trigger welfare_fraud_pipeline    # start a run now
docker compose exec airflow-scheduler airflow dags pause welfare_fraud_pipeline      # stop the schedule firing
docker compose exec airflow-scheduler airflow dags unpause welfare_fraud_pipeline    # resume the schedule
docker compose exec airflow-scheduler airflow dags next-execution welfare_fraud_pipeline   # when the next scheduled run fires

# Inspect runs
docker compose exec airflow-scheduler airflow dags list-runs -d welfare_fraud_pipeline

# Retry after a fix
docker compose exec airflow-scheduler airflow tasks clear welfare_fraud_pipeline --only-failed --yes
```

- **`trigger`** starts a manual run immediately, independent of the schedule.
- **`pause` / `unpause`** control the *schedule* only. A paused DAG can still be triggered manually.
- **`tasks clear --only-failed --yes`** resets failed tasks (and the ones blocked behind them) so the scheduler
  re-runs them inside the same run, without redoing what already succeeded. `--yes` skips the confirmation prompt.

You can do all of this in the UI too (toggle = pause/unpause, play button = trigger, click a task then **Clear**).

---

## 6. Checking that a run actually worked

Green tasks only mean nothing crashed. The final task `validate_run` fails the run if a table is empty,
the result files are stale, or a detector falls below its recall/precision floor on the planted answer key.
You can also check by hand:

```bash
ls -l data/results/                                                                   # timestamps should be from the latest run
sed -n '/## Coverage matrix/,/## Method scorecard/p' data/results/comparison_report.md   # recall by method and pattern
```

Reading a task's log when it fails (from the host, no Docker needed):
```bash
ls airflow/logs/dag_id=welfare_fraud_pipeline/
tail -n 40 airflow/logs/dag_id=welfare_fraud_pipeline/run_id=*/task_id=rule_based/attempt=*.log
```
Replace `rule_based` with the failing task's name. The same log is under the task's **Logs** tab in the UI.

---

## 7. Schedule and presentations

The DAG runs daily at 21:00 Lahore time (`SCHEDULE_CRON` and `TIMEZONE` at the top of the DAG file).
Missed runs are skipped, not backfilled (`catchup=False`), and the laptop must be awake for it to fire.

Every run **truncates and reloads** the project tables. Before a presentation or demo:
```bash
docker compose exec airflow-scheduler airflow dags pause welfare_fraud_pipeline
```
and afterwards:
```bash
docker compose exec airflow-scheduler airflow dags unpause welfare_fraud_pipeline
```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| DAG not in the UI | Folder named `dag` instead of `dags`, or Airflow hasn't rescanned | Check `ls airflow`; run `airflow dags reserialize`; refresh; clear UI filters |
| `list-import-errors` shows an error | Python error in the DAG file | Read the message; fix the file |
| Run stays **queued**, no task starts | Scheduler or worker not healthy | `docker compose ps`, then `docker compose logs --tail=40 airflow-scheduler`. Logs mentioning `host.docker.internal:6379` mean `DB_HOST` was set in the compose file: rename it to `PROJECT_DB_HOST` and `docker compose up -d` |
| Extract/transform/load fail | Data files or project DB problem | Open the task log; check `data/raw/` exists and Postgres is running on the Mac |
| Detector tasks fail in ~1-2 seconds | Can't reach the DB, or a wrong module path | Read `airflow/logs/.../task_id=<name>/attempt=*.log`. `No module named ...` = module path constants at the top of the DAG file are wrong |
| `connection refused` to the database | Postgres not running, or not accepting connections from Docker | Start Postgres on the Mac; re-run the DB smoke test (2.9) |
| Port 8080 already in use | Another project's Airflow is running | `docker ps`, then `docker compose down` in that other project's folder |
| `docker: command not found` / cannot connect to daemon | Docker Desktop is not running | Open Docker Desktop and wait for it to say "running" |
| Code change in `src/` has no effect | Very rarely a stale process | `docker compose restart airflow-worker` |
| `deps ok` fails after editing requirements | Image not rebuilt | `docker compose build` then `docker compose up -d` |

---

## 9. Quick reference

```bash
docker compose up -d                                   # start everything
docker compose ps                                      # health check
docker compose logs --tail=50 airflow-scheduler        # scheduler logs
docker compose stop                                    # pause everything (keeps state)
docker compose down                                    # shut down cleanly
docker compose build && docker compose up -d           # apply Dockerfile / requirements changes
docker compose down --volumes --remove-orphans         # full reset of Airflow (not your data)

docker compose exec airflow-scheduler airflow dags trigger welfare_fraud_pipeline
docker compose exec airflow-scheduler airflow dags pause welfare_fraud_pipeline
docker compose exec airflow-scheduler airflow dags unpause welfare_fraud_pipeline
docker compose exec airflow-scheduler airflow tasks clear welfare_fraud_pipeline --only-failed --yes
```

UI: http://localhost:8080 (login `airflow` / `airflow`)