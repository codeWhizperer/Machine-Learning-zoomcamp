FROM python:3.12-slim

WORKDIR /app

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipenv files first (for better Docker layer caching)
COPY Pipfile Pipfile.lock ./

# Install dependencies from Pipenv into the system environment
RUN pipenv install --system --deploy

# Copy the rest of your project
COPY . .

# Expose the port Gunicorn will run on
EXPOSE 9696

# Start your Flask app via Gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
