# Security policy

## Reporting a vulnerability

Please **do not** open a public issue. Use GitHub's private vulnerability reporting
(**Security → Report a vulnerability** on the repository page). You should get an
acknowledgement within 7 days. Include affected versions, reproduction steps and impact.

## Supported versions

Only the latest release receives fixes while the project is in 0.x.

## Security properties of the SDK

- The API never accepts prompt or completion text. User-supplied `attributes` are
  exported as given, so do not pass sensitive data in them.
- Request IDs are span attributes, never metric labels. Metric labels are capped
  per key.
- OTLP endpoints containing credentials, query strings or control characters are
  rejected. Use `OTEL_EXPORTER_OTLP_HEADERS` for authentication headers and HTTPS for
  remote collectors.
- Exporter failures and timeouts are contained. Shutdown is bounded.

## The local stack is for development only

`docker-compose.yml` binds every port to `127.0.0.1`, enables anonymous Grafana viewer
access, and uses the default admin password `admin` unless `GRAFANA_ADMIN_PASSWORD` is
set. Do not deploy it on a reachable network as is.
