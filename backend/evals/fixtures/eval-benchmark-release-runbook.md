# Benchmark Release Runbook

## Pre-release Checklist

- Confirm the target branch has passed automated tests.
- Verify that database migrations have a documented rollback plan.
- Announce the release window in the engineering channel.

## Release Execution

- Deploy during the approved release window.
- Run smoke tests against `/health`, `/docs`, and one representative `/chat` request.
- Record the deployed version and operator in the release notes.

## Incident Handling

- If a release blocks a critical user workflow, pause rollout and open an urgent support ticket.
- The release owner coordinates rollback approval with the incident commander.
- After rollback, rerun smoke tests and attach the incident summary to the release notes.

