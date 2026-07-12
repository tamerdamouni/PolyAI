# PolyAI

## Setup

Create and activate a virtual environment from the repo root directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now show `(.venv)`. Keep this environment active whenever you run any service.

See each service's README for how to configure and run it.

## Running locally

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

`docker-compose.yml` only references pre-built images — it is what the servers run, and it
deliberately contains no build instructions so a server can never build from source. The
`docker-compose.build.yml` overlay adds the build contexts back for local development.

## Deployment

Every push to `dev` or `main` builds a Docker image, pushes it to DockerHub, and redeploys the
affected service. Nothing is built on the server and no source code is deployed there — the stack
runs entirely from pre-built images via `docker compose up -d`.

### Workflows

Each service has its own workflow, scoped by a path filter, so a commit touching only the agent
rebuilds and redeploys only the agent:

| Workflow | Triggered by | DockerHub image |
| --- | --- | --- |
| `deploy-agent.yaml` | `services/agent/**` | `<user>/polyai-agent` |
| `deploy-yolo.yaml` | `services/yolo/**` | `<user>/polyai-yolo` |
| `deploy-frontend.yaml` | `services/frontend/**` | `<user>/polyai-frontend` |
| `deploy-stack.yaml` | `docker-compose.yml`, `services/prometheus/**` | none (no build job) |

The branch selects the target instance: `dev` deploys to the dev instance, `main` to prod.

### Image tags

Images are tagged `<branch>-<commit-sha>` (for example `dev-a1b2c3d…`). Tags are immutable and
unique per build; nothing is ever tagged `latest`. The branch prefix matters because the frontend
bakes `NEXT_PUBLIC_AGENT_URL` into its bundle at build time, so a dev-built frontend image must
never be reused in prod even if the two branches point at the same commit.

Each deploy writes its tag into `/home/ubuntu/polyai/.env` on the server, updating only its own
key:

```
DOCKERHUB_USERNAME=<user>
AGENT_IMAGE_TAG=dev-a1b2c3d
YOLO_IMAGE_TAG=dev-9f8e7d6
FRONTEND_IMAGE_TAG=dev-a1b2c3d
```

`docker-compose.yml` substitutes these variables into the image names, so services stay on their
own tags independently.

**To roll back**, edit the relevant `*_IMAGE_TAG` in that `.env` to an earlier tag and run
`docker compose up -d <service>`, or re-run the older GitHub Actions workflow run.

### Required GitHub secrets

`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `DEV_INSTANCE_IP`, `DEV_INSTANCE_SSH_KEY`,
`PROD_INSTANCE_IP`, `PROD_INSTANCE_SSH_KEY`.

### Server bootstrap (one time, per instance)

The instance needs Docker with the Compose plugin, and:

```bash
mkdir -p /home/ubuntu/polyai/services/agent /home/ubuntu/polyai/services/yolo

# Secret env files, server-managed. CI never writes these.
# Populate from each service's .env.example.
vi /home/ubuntu/polyai/services/agent/.env
vi /home/ubuntu/polyai/services/yolo/.env

# Remove the old deployment: source checkout and systemd units.
sudo systemctl disable --now yolo.service agent.service frontend.service || true
rm -rf /home/ubuntu/PolyAI
```

`docker-compose.yml`, `prometheus.yml`, and the image-tag `.env` are all created by the first
deploy, so no further setup is needed.