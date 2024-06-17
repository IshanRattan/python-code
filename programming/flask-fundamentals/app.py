from flask import Flask, jsonify, request
from flask_caching import Cache
from flask_compress import Compress
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import time
import random
import logging
import os

app = Flask(__name__)
app.config.from_object("config.Config")

# Enable response compression
Compress(app)

# Set up API rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Set up caching
cache = Cache(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route("/")
def root():
    app.logger.info("Root endpoint accessed")
    return jsonify(message="Welcome! Flask v2 up and running with CI/CD.")

@app.route("/health")
def health_check():
    return jsonify(status="OK"), 200

@app.route("/heavy")
def heavy_task():
    app.logger.info("Heavy endpoint - simulating workload")
    time.sleep(65)
    return jsonify(result="Task completed after heavy load.")

@app.route("/cacheme/<param>")
@cache.cached(timeout=120)
def cache_response(param):
    time.sleep(20)
    process_id = os.getpid()
    app.logger.info(f"Cache used for param: {param} (PID: {process_id})")
    return jsonify(
        result=f"Processed: {param}",
        random=random.randint(1, 1000),
        worker=process_id
    )

@app.route("/bigjson")
def large_json():
    # Create a large JSON array for testing purposes
    data = [{"item": i, "value": "x" * 100} for i in range(2000)]
    return jsonify(data)

@app.route('/api')
@limiter.limit("5/minute")
def limited_api():
    process_id = os.getpid()
    app.logger.info(f"API rate-limited call (PID: {process_id})")
    return "success"

@app.route("/error")
def throw_error():
    app.logger.error("Test error triggered for monitoring")
    raise Exception("Generated test exception for error tracking")

# Central error handling
@app.errorhandler(Exception)
def error_handler(e):
    app.logger.exception("Exception caught in global handler")
    return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
