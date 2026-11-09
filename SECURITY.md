# Security

Please report a security problem privately through GitHub: on the repository page, open **Security**, then **Report a vulnerability**. If that button is missing, open an issue that asks for a private contact, without details of the problem. Do not put exploit details in a public issue.

The API has no authentication. `up` binds it to 127.0.0.1 by default; the container image, the compose file, and the Kubernetes Deployment bind it to 0.0.0.0 inside their networks (`Dockerfile`, `docker-compose.yml`, `deploy/k8s/deployment.yaml`). Do not expose them to an untrusted network.
