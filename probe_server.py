import argparse
import socket


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe a TCP server to inspect its initial protocol/messages."
    )
    parser.add_argument("--host", default="10.245.30.229", help="Server host/IP")
    parser.add_argument("--port", type=int, default=50555, help="Server port")
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Socket timeout in seconds",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        print(f"Connected to {args.host}:{args.port}")
        sock.settimeout(args.timeout)

        try:
            data = sock.recv(4096)
            if data:
                print("\nReceived first bytes:")
                print(data)
                try:
                    print("\nDecoded as UTF-8:")
                    print(data.decode("utf-8", errors="replace"))
                except Exception:
                    pass
            else:
                print("Connected, but server sent nothing immediately.")
        except socket.timeout:
            print("Connected, but no immediate data from server.")

        print("\nType a line to send it to the server. Type 'quit' to exit.")
        while True:
            try:
                msg = input("Send> ")
            except EOFError:
                break

            if msg.lower() in {"quit", "exit"}:
                break

            sock.sendall(msg.encode("utf-8") + b"\n")
            try:
                data = sock.recv(4096)
                if data:
                    print("\nReply bytes:")
                    print(data)
                    print("\nReply decoded:")
                    print(data.decode("utf-8", errors="replace"))
                else:
                    print("Server closed the connection.")
                    break
            except socket.timeout:
                print("No reply before timeout.")


if __name__ == "__main__":
    main()
