#!/usr/bin/env python3
"""
Fixed Test Server (For Your Laptop)
====================================

This is an improved version that handles large data transfers more reliably.

Usage:
    python3 test_server_laptop_fixed.py --port 8889
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
            # Increase socket buffer sizes for large data transfers
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)  # 1MB receive buffer
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)  # 1MB send buffer
            
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
                    
                    # Set socket options for the client connection
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)
                    client_socket.settimeout(30.0)  # 30 second timeout
                    
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
    
    def recv_exact(self, sock, length):
        """Receive exactly 'length' bytes with progress tracking"""
        data = b""
        bytes_received = 0
        chunk_size = 4096  # Receive in 4KB chunks
        
        print(f"📥 Receiving {length} bytes...")
        
        while bytes_received < length:
            try:
                remaining = length - bytes_received
                to_receive = min(chunk_size, remaining)
                chunk = sock.recv(to_receive)
                
                if not chunk:
                    print(f"❌ Connection closed by client. Received {bytes_received}/{length} bytes")
                    return None
                    
                data += chunk
                bytes_received += len(chunk)
                
                # Show progress for large transfers
                if length > 10000 and bytes_received % 50000 == 0:
                    progress = (bytes_received / length) * 100
                    print(f"   📊 Progress: {bytes_received}/{length} bytes ({progress:.1f}%)")
                
            except socket.timeout:
                print(f"⏰ Timeout while receiving data. Got {bytes_received}/{length} bytes")
                return None
            except Exception as e:
                print(f"❌ Error receiving data: {e}")
                return None
        
        print(f"✅ Received all {length} bytes successfully")
        return data
            
    def handle_client(self, client_socket, client_address):
        """Handle client connection"""
        try:
            print(f"📡 Handling client {client_address}")
            
            # Receive data length (4 bytes)
            length_data = self.recv_exact(client_socket, 4)
            if not length_data:
                print(f"❌ Failed to receive length data from {client_address}")
                return
                
            data_length = int.from_bytes(length_data, byteorder='big')
            print(f"📊 Expecting {data_length} bytes from {client_address}")
            
            # Receive the actual data
            received_data = self.recv_exact(client_socket, data_length)
            if not received_data:
                print(f"❌ Failed to receive complete data from {client_address}")
                return
            
            # Try to decode as JSON first (text message)
            try:
                message = json.loads(received_data.decode())
                print(f"📝 Text message from {client_address}: {message}")
                
                # Send response
                response = {"status": "received", "message": "Hello from laptop!", "timestamp": time.time()}
                response_data = json.dumps(response).encode()
                
                # Send response length then data
                client_socket.send(len(response_data).to_bytes(4, byteorder='big'))
                client_socket.send(response_data)
                print(f"✅ Response sent to {client_address}")
                
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
                print(f"✅ Response sent to {client_address}")
                
        except Exception as e:
            print(f"❌ Error handling client {client_address}: {e}")
            
        finally:
            client_socket.close()
            print(f"🔌 Connection closed: {client_address}")

def main():
    print("💻 Fixed Test Server - Laptop Side")
    print("========================================")
    print("This server handles large data transfers more reliably")
    
    parser = argparse.ArgumentParser(description='Fixed test server for network communication')
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