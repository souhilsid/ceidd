# port_checker.py
import socket
import subprocess
import sys
import os


def _platform_urls(port: int, local_ip: str):
    return [
        f"http://localhost:{port}",
        f"http://127.0.0.1:{port}",
        f"http://{local_ip}:{port}",
        f"http://0.0.0.0:{port}",
    ]

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
    """Kill process using the port on Windows, Linux, or macOS."""
    try:
        if os.name == 'nt':
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                capture_output=True,
                text=True,
                shell=True,
            )

            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'LISTENING' in line:
                        parts = line.split()
                        pid = parts[-1]
                        print(f"Killing process {pid} on port {port}")
                        subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
                        return True
        else:
            finder = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            pid = finder.stdout.strip().splitlines()
            if pid:
                for entry in pid:
                    print(f"Killing process {entry} on port {port}")
                    subprocess.run(["kill", "-9", entry], check=False)
                return True
        return False
    except Exception as e:
        print(f"Error killing process: {e}")
        return False

def main():
    port = int(os.getenv("PORT", "8501"))
    print(" Port Troubleshooter")
    print("=" * 50)
    
    # Check current port
    available, message = check_port(port)
    print(f"Port {port}: {message}")
    
    if not available:
        print(f"\n Port {port} is busy. Trying to fix...")
        
        # Try to kill the process
        if kill_process_on_port(port):
            print(f" Killed process on port {port}")
        else:
            print(" Could not kill process automatically")
        
        # Find alternative port
        new_port, msg = find_available_port(port)
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
        for url in _platform_urls(port, local_ip):
            print(url)
        
    except Exception as e:
        print(f"Error getting network info: {e}")

if __name__ == "__main__":
    main()