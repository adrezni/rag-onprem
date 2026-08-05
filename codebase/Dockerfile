FROM registry.access.redhat.com/ubi9/python-312:latest
   WORKDIR /opt/app-root/src
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]