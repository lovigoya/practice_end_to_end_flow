FROM python

# Copy the application code to the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Expose a port to access the application
EXPOSE 5000

# Set the command to run the application
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
