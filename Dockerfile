# Use an official Python base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from the host codeforces folder into /app
COPY . .

# Run the final model script
CMD ["python", "final_model.py"]
