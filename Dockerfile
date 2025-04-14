FROM python:3.11-slim

# Set the working directory
WORKDIR /app

COPY . /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# run streamlit app
CMD ["streamlit", "app.py"]