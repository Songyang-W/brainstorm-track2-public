#!/usr/bin/env python3
"""
Control Client for Track 2 Streaming Server

A simple terminal program that connects to the /control WebSocket endpoint
and sends arrow key presses as control messages to move the array position.
"""

import asyncio
import json
import sys

import typer
import websockets
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(help="Control the streaming server array position with arrow keys")
console = Console()


class ControlClient:
    """Client for sending control messages to the streaming server."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.url = f"ws://{host}:{port}/control"
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.running = False
        self.connected = False
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self) -> None:
        """Connect to the control endpoint."""
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True

            # Wait for acknowledgment
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                data = json.loads(message)
                if data.get("type") == "control_ack":
                    console.print(f"[green]✓ Connected to {self.url}[/green]")
            except asyncio.TimeoutError:
                console.print(
                    f"[yellow]Connected to {self.url} (no acknowledgment received)[/yellow]"
                )
            except Exception:
                pass

        except ConnectionRefusedError:
            console.print(f"[red]✗ Error: Could not connect to {self.url}[/red]")
            console.print("[dim]Make sure the streaming server is running[/dim]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]✗ Connection error: {e}[/red]")
            sys.exit(1)

    async def send_key_event(self, key: str, pressed: bool) -> None:
        """Send a key press/release event to the server."""
        if not self.connected or self.websocket is None:
            return

        message = {
            "type": "key",
            "key": key,
            "pressed": pressed,
        }
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            console.print(f"[red]Error sending message: {e}[/red]")
            self.connected = False

    async def send_position(self, x: float, y: float) -> None:
        """Send a direct position update to the server."""
        if not self.connected or self.websocket is None:
            return

        message = {
            "type": "position",
            "x": x,
            "y": y,
        }
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            console.print(f"[red]Error sending message: {e}[/red]")
            self.connected = False

    def on_key_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key press events."""
        if self.loop is None:
            return
        try:
            if key == keyboard.Key.up:
                asyncio.run_coroutine_threadsafe(
                    self.send_key_event("up", True), self.loop
                )
            elif key == keyboard.Key.down:
                asyncio.run_coroutine_threadsafe(
                    self.send_key_event("down", True), self.loop
                )
            elif key == keyboard.Key.left:
                asyncio.run_coroutine_threadsafe(
                    self.send_key_event("left", True), self.loop
                )
            elif key == keyboard.Key.right:
                asyncio.run_coroutine_threadsafe(
                    self.send_key_event("right", True), self.loop
                )
        except AttributeError:
            pass

    def on_key_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key release events."""
        if self.loop is None:
            return
        try:
            if key == keyboard.Key.up:
                asyncio.run_coroutine_threadsafe(
                    self.send_key_event("up", False), self.loop
                )
            elif key == keyboard.Key.down:
                asyncio.run_coroutine_threadsafe(
                    self.send_key_event("down", False), self.loop
                )
            elif key == keyboard.Key.left:
                asyncio.run_coroutine_threadsafe(
                    self.send_key_event("left", False), self.loop
                )
            elif key == keyboard.Key.right:
                asyncio.run_coroutine_threadsafe(
                    self.send_key_event("right", False), self.loop
                )
        except AttributeError:
            pass

    async def run(self) -> None:
        """Run the control client."""
        # Store event loop for thread-safe async calls
        self.loop = asyncio.get_event_loop()

        await self.connect()

        # Create status panel
        status_text = Text()
        status_text.append("Arrow Keys: ", style="bold cyan")
        status_text.append("↑ ↓ ← →", style="bold")
        status_text.append(" to move array position\n", style="dim")
        status_text.append("Press ", style="dim")
        status_text.append("Ctrl+C", style="bold")
        status_text.append(" to exit", style="dim")

        console.print()
        console.print(
            Panel(
                status_text,
                title="[bold cyan]Control Client[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print()
        console.print(
            "[dim]Click away from this terminal to keep the display clean![/dim]"
        )
        console.print(
            "[dim]Arrow keys will work globally to control the array position.[/dim]"
        )
        console.print()

        # Setup keyboard listener
        listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release,
        )
        listener.start()

        self.running = True

        try:
            # Keep connection alive and handle any incoming messages
            while self.running:
                try:
                    if self.websocket:
                        # Wait for messages (with timeout to allow checking running state)
                        try:
                            await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                            # Silently handle any messages from server
                        except asyncio.TimeoutError:
                            # Timeout is fine, just check if we should continue
                            pass
                except websockets.exceptions.ConnectionClosed:
                    console.print("\n[yellow]Connection closed by server[/yellow]")
                    break
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        finally:
            self.running = False
            listener.stop()
            if self.websocket:
                await self.websocket.close()


@app.command()
def main(
    host: str = typer.Option("localhost", help="Server hostname"),
    port: int = typer.Option(8765, help="Server port"),
) -> None:
    """Run the control client."""
    client = ControlClient(host, port)
    asyncio.run(client.run())


if __name__ == "__main__":
    app()
