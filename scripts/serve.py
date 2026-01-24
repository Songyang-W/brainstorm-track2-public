"""
Web server for the Neural Data Viewer.

This server serves the static web files. Web clients connect directly
to the data stream (stream_data.py) instead of through this server.
"""

import asyncio
from pathlib import Path

import typer
from aiohttp import web
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

VIEWER_DIR = Path(__file__).parent.parent / "example_app"

app = typer.Typer(help="Neural Data Viewer Web Server")
console = Console()


class WebViewerServer:
    """Web server that serves static files for the neural data viewer."""

    def __init__(
        self,
        host: str,
        port: int,
    ):
        self.host = host
        self.port = port

    async def index_handler(self, request: web.Request) -> web.FileResponse:
        """Serve index.html."""
        return web.FileResponse(VIEWER_DIR / "index.html")

    def create_app(self) -> web.Application:
        """Create the web application."""
        app = web.Application()

        # Serve index.html at root
        app.router.add_get("/", self.index_handler)

        # Serve static files
        app.router.add_static("/", VIEWER_DIR, show_index=True)

        return app

    async def start(self) -> None:
        """Start the server."""
        # Create web app
        app = self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        # Display server info
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("[cyan]Web Viewer[/cyan]", f"http://{self.host}:{self.port}")

        console.print(
            Panel(
                table,
                title="[bold green]🚀 Server Running[/bold green]",
                border_style="green",
            )
        )
        console.print()
        console.print(
            "[dim]Note: Web clients connect directly to the data stream at ws://localhost:8765[/dim]"
        )
        console.print()

        # Keep server running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass

    def stop(self) -> None:
        """Stop the server."""
        pass


@app.command()
def main(
    port: int = typer.Option(8000, help="Web server port"),
    host: str = typer.Option("localhost", help="Host to bind to"),
) -> None:
    """
    Start the Neural Data Viewer web server.

    This server serves static files. Web clients connect directly
    to the data stream (ws://localhost:8765) instead of through this server.
    """
    # Print header
    console.print()
    console.print(
        Panel(
            "[bold cyan]Neural Data Viewer Server[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    # Print configuration
    config_table = Table(show_header=False, box=None, padding=(0, 1))
    config_table.add_row("[dim]Web Server[/dim]", f"[cyan]http://{host}:{port}[/cyan]")
    config_table.add_row(
        "[dim]Data Stream[/dim]", "[cyan]ws://localhost:8765[/cyan] (direct connection)"
    )

    console.print(config_table)
    console.print()

    server = WebViewerServer(
        host=host,
        port=port,
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]⚠[/yellow] Shutting down...")
        server.stop()


if __name__ == "__main__":
    app()
