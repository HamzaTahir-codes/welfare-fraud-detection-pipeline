# Notices

Unless stated otherwise below, everything in this repository is licensed under the MIT License (see [`LICENSE`](LICENSE)).

## Third-party material

| File | Source | License |
|---|---|---|
| `docker-compose.yaml` | Adapted from the official Apache Airflow 2.10.5 `docker-compose.yaml` ([airflow.apache.org](https://airflow.apache.org/docs/apache-airflow/2.10.5/docker-compose.yaml)). Copyright The Apache Software Foundation. | Apache License 2.0 (see [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0)) |

Changes to `docker-compose.yaml`: bind-mounts the project into the containers, sets `PROJECT_ROOT`, `PYTHONPATH` and
`PROJECT_DB_HOST`, adds a `host-gateway` entry so tasks can reach the host's PostgreSQL, and builds the image from the
project `Dockerfile`.

## Dependencies

Python packages listed in `requirements*.txt` are installed from PyPI and are not redistributed here; each is
covered by its own license.

## Data

All data in this repository is synthetic, generated with [Faker](https://github.com/joke2k/faker). It contains no
real personal information; any resemblance of generated names, ID numbers or account numbers to real ones is
coincidental.
