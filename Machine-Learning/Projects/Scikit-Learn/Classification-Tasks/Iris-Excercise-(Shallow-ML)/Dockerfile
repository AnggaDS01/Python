# Gunakan image base resmi Python
FROM python:3.9-slim

# Set lingkungan kerja di dalam container
WORKDIR /app

# Copy semua file ke dalam container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port yang akan digunakan oleh Flask
EXPOSE 8000

# Perintah untuk menjalankan aplikasi Flask
CMD ["python", "app.py"]