# port_checker.py
import socket
import subprocess
import sys
import os

def check_port(port=8501):
    """Check if port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return True, f"Port {port} is available"
        except OSError as e:
            return False, f"Port {port} is in use: {e}"

def find_available_port(start_port=8501, max_attempts=10):
    """Find an available port"""
    for port in range(start_port, start_port + max_attempts):
        available, message = check_port(port)
        if available:
            return port, message
    return None, "No available ports found"

def kill_process_on_port(port=8501):
    """Kill process using the port (Windows)"""
    try:
        # Find PID using the port
        result = subprocess.run(
            ["netstat", "-ano", "|", "findstr", f":{port}"],
            capture_output=True, text=True, shell=True
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'LISTENING' in line:
                    parts = line.split()
                    pid = parts[-1]
                    print(f"Killing process {pid} on port {port}")
                    subprocess.run(["taskkill", "/PID", pid, "/F"])
                    return True
        return False
    except Exception as e:
        print(f"Error killing process: {e}")
        return False

def main():
    print(" Port Troubleshooter")
    print("=" * 50)
    
    # Check current port
    available, message = check_port(8501)
    print(f"Port 8501: {message}")
    
    if not available:
        print("\n Port 8501 is busy. Trying to fix...")
        
        # Try to kill the process
        if kill_process_on_port(8501):
            print(" Killed process on port 8501")
        else:
            print(" Could not kill process automatically")
        
        # Find alternative port
        new_port, msg = find_available_port(8501)
        print(f"\n Alternative: Use port {new_port}")
        print(f"   Run: streamlit run app.py --server.port {new_port}")
    
    # Show network information
    print("\n Network Information:")
    try:
        # Get local IP
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Local IP: {local_ip}")
        
        # Test URLs
        print("\n Try these URLs:")
        print(f"http://localhost:8501")
        print(f"http://127.0.0.1:8501") 
        print(f"http://{local_ip}:8501")
        print(f"http://0.0.0.0:8501")
        
    except Exception as e:
        print(f"Error getting network info: {e}")

if __name__ == "__main__":
    main()