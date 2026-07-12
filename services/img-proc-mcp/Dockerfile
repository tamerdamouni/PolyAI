FROM python:3.12-slim-trixie

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 9000

CMD ["python", "app.py"]
