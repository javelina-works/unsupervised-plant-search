FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by Rasterio and other libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libexpat1 \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app
COPY input/ /app/input/


# Expose port 5006
EXPOSE 5006

# Set the command to run the Bokeh server
CMD ["bokeh", "serve", "bokeh-int.py", "--port", "5006", "--address", "0.0.0.0", "--allow-websocket-origin", "localhost:5006", "--allow-websocket-origin", "unsupervised-plant-search-production.up.railway.app", "--websocket-max-message-size=250000000"]
