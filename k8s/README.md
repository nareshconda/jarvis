# Running Jarvis on k3s (optional)

Example manifests for a single-node k3s, for example on a Raspberry Pi 5.

1. Build the image and push it to a local registry:
   `docker build -t localhost:5000/jarvis-server:latest . && docker push localhost:5000/jarvis-server:latest`
2. Replace `YOUR_USER`, `/mnt/data` and `jarvis.example.com` in the YAML files.
3. Provide `.env` to the pod, either in the mounted code folder or as a secret:
   `kubectl -n jarvis-server create secret generic jarvis-env --from-env-file=.env`,
   then use `envFrom`.
4. Apply the manifests: `kubectl apply -f k8s/`

`hostNetwork: true` lets Jarvis ping your PCs and send Wake-on-LAN. `/dev/snd` is only
needed for the server's own microphone or speaker.
