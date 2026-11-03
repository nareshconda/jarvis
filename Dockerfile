FROM python:3.11-slim

WORKDIR /app

# portaudio/alsa: voice loop · ffmpeg: audio+video · ping/ssh: device checks
# plocate: file search · ngrok: optional WhatsApp tunnel
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ffmpeg curl git iputils-ping openssh-client plocate \
        portaudio19-dev libasound2-dev libportaudiocpp0 ca-certificates gnupg \
    && curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc > /etc/apt/trusted.gpg.d/ngrok.asc \
    && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" > /etc/apt/sources.list.d/ngrok.list \
    && apt-get update && apt-get install -y --no-install-recommends ngrok \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5006 5446 5010
CMD ["python", "jarvis_server.py"]
