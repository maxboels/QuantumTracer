#!/usr/bin/env python3
"""
Simple Test Server (For Your Laptop)
====================================

This is a minimal test server to verify network communication works
before testing the full inference system.

Usage:
    python3 test_server_laptop.py --port 8889
"""

import argparse
import socket
import json
import threading
import time
import struct
import cv2
import numpy as np

class TestServer:
    def __init__(self, port=8889):
        self.port = port
        self.running = False
        
    def get_local_ip(self):
        """Get the local IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "localhost"
        
    def start_server(self):
        """Start the test server"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.port))
            server_socket.listen(5)
            
            print(f"🚀 Test server started on port {self.port}")
            print(f"💻 Laptop IP: {self.get_local_ip()}")
            print("Waiting for connections from Raspberry Pi...")
            print("=" * 50)
            
            self.running = True
            
            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()
                    print(f"🔗 New connection from {client_address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except KeyboardInterrupt:
                    print("\n🛑 Server stopping...")
                    break
                except Exception as e:
                    print(f"❌ Error accepting connection: {e}")
                    
            server_socket.close()
            self.running = False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            
    def handle_client(self, client_socket, client_address):
        """Handle client connection"""
        try:
            print(f"📡 Handling client {client_address}")
            
            # Receive data length
            length_data = client_socket.recv(4)
            if len(length_data) != 4:
                print(f"❌ Invalid length data from {client_address}")
                return
                
            data_length = int.from_bytes(length_data, byteorder='big')
            print(f"📊 Expecting {data_length} bytes from {client_address}")
            
            # Receive the actual data
            received_data = b""
            while len(received_data) < data_length:
                chunk = client_socket.recv(min(4096, data_length - len(received_data)))
                if not chunk:
                    break
                received_data += chunk
                
            if len(received_data) != data_length:
                print(f"❌ Incomplete data from {client_address}: got {len(received_data)}, expected {data_length}")
                return
            
            # Try to decode as JSON first (text message)
            try:
                message = json.loads(received_data.decode())
                print(f"📝 Text message from {client_address}: {message}")
                
                # Send response
                response = {"status": "received", "message": "Hello from laptop!", "timestamp": time.time()}
                response_data = json.dumps(response).encode()
                client_socket.send(len(response_data).to_bytes(4, byteorder='big'))
                client_socket.send(response_data)
                
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Assume it's image data
                print(f"📸 Image data from {client_address}: {len(received_data)} bytes")
                
                try:
                    # Try to decode as image
                    nparr = np.frombuffer(received_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        print(f"✅ Valid image received: {img.shape}")
                        
                        # Save test image
                        timestamp = int(time.time())
                        filename = f"test_frame_{timestamp}.jpg"
                        cv2.imwrite(filename, img)
                        print(f"💾 Saved test image: {filename}")
                        
                        # Send response
                        response_text = f"Image received successfully! Size: {img.shape}, saved as {filename}"
                    else:
                        response_text = "Invalid image data received"
                        print("❌ Invalid image data")
                        
                except Exception as e:
                    response_text = f"Error processing image: {e}"
                    print(f"❌ Image processing error: {e}")
                
                # Send text response
                response_data = response_text.encode()
                client_socket.send(len(response_data).to_bytes(4, byteorder='big'))
                client_socket.send(response_data)
                
        except Exception as e:
            print(f"❌ Error handling client {client_address}: {e}")
            
        finally:
            client_socket.close()
            print(f"🔌 Connection closed: {client_address}")

def main():
    print("💻 Simple Test Server - Laptop Side")
    print("========================================")
    print("This server will receive test messages from your Raspberry Pi")
    
    parser = argparse.ArgumentParser(description='Test server for network communication')
    parser.add_argument('--port', type=int, default=8889, help='Port to listen on')
    args = parser.parse_args()
    
    print(f"Port: {args.port}")
    print()
    
    server = TestServer(args.port)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == '__main__':
    main()