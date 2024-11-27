FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["bokeh", "serve", "bokeh-int.py", "--port", "5006", "--address", "0.0.0.0"]
