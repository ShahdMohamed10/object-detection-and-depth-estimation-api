"""
Run the pickle-optimized API - this script demonstrates both first-time loading
(which will create pickle files) and subsequent faster loading times.
"""

import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting the pickle-optimized API")
    
    # Record start time to measure initialization time
    start_time = time.time()
    
    # Import and run the API
    from pickle_optimized_api import app
    
    # Calculate and log the initialization time
    init_time = time.time() - start_time
    logger.info(f"API initialized in {init_time:.2f} seconds")
    
    # Check if pickle files were created
    pickle_dir = "pickled_models"
    if os.path.exists(pickle_dir):
        pickle_files = os.listdir(pickle_dir)
        logger.info(f"Pickle directory contains: {pickle_files}")
        
        # Get the total size of pickle files
        total_size = sum(os.path.getsize(os.path.join(pickle_dir, f)) for f in pickle_files)
        logger.info(f"Total size of pickle files: {total_size / (1024*1024):.2f} MB")
    
    # Run the Flask app
    logger.info("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True) 