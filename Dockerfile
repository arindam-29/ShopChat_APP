FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]

# Build: docker build -t streamlit:1.0 .
# Check: docker images
# Run:   docker run -p 8501:8501 streamlit:1.0