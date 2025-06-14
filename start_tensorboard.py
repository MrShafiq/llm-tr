import os
import subprocess
import logging
import sys
from src.utils.config import TENSORBOARD_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tensorboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_tensorboard_path():
    """Get the full path to the TensorBoard executable"""
    # Get Python executable directory
    python_dir = os.path.dirname(sys.executable)
    
    # Try different possible locations
    possible_paths = [
        os.path.join(python_dir, 'Scripts', 'tensorboard.exe'),  # Windows
        os.path.join(python_dir, 'Scripts', 'tensorboard'),      # Unix
        os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Python', 'Python310', 'Scripts', 'tensorboard.exe'),  # Windows user install
        os.path.join(os.path.expanduser('~'), '.local', 'bin', 'tensorboard')  # Unix user install
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found TensorBoard at: {path}")
            return path
    
    raise FileNotFoundError("Could not find TensorBoard executable. Please ensure TensorBoard is installed.")

def start_tensorboard():
    """Start TensorBoard server"""
    try:
        # Ensure the log directory exists
        os.makedirs(TENSORBOARD_DIR, exist_ok=True)
        
        # Get TensorBoard path
        tensorboard_path = get_tensorboard_path()
        
        # Start TensorBoard
        logger.info(f"Starting TensorBoard with log directory: {TENSORBOARD_DIR}")
        tensorboard_cmd = [
            tensorboard_path,
            '--logdir', TENSORBOARD_DIR,
            '--port', '6006',
            '--bind_all'
        ]
        
        # Run TensorBoard in a subprocess
        process = subprocess.Popen(
            tensorboard_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info("TensorBoard started successfully")
        logger.info("Access TensorBoard at http://localhost:6006")
        
        # Monitor the process
        while True:
            output = process.stdout.readline()
            if output:
                logger.info(output.strip())
            
            error = process.stderr.readline()
            if error:
                logger.error(error.strip())
            
            # Check if process has ended
            if process.poll() is not None:
                break
        
    except Exception as e:
        logger.error(f"Error starting TensorBoard: {e}")
        raise

if __name__ == "__main__":
    start_tensorboard() 