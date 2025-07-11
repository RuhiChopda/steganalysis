# Use an older compatible Python base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy your files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
