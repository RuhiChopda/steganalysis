# Use a compatible Python version
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Start the Flask app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
