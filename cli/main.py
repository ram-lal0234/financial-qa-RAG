"""
Financial Q&A CLI Chatbot — Part 1

Usage:
    python -m cli.main
    python -m cli.main --no-stream
    python -m cli.main --model gpt-4o
"""

from __future__ import annotations

import sys
import typer
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from rich import print as rprint

from core.llm import LLMClient, ConversationHistory
from core.config import settings

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="financial-qa",
    help="Financial Earnings Call Q&A Assistant",
    add_completion=False,
)
console = Console()

WELCOME_BANNER = """
[bold cyan]Financial Q&A Assistant[/bold cyan]
[dim]Powered by {model} · Type [bold]/help[/bold] for commands · [bold]/quit[/bold] to exit[/dim]
"""

HELP_TEXT = """
[bold]Available Commands[/bold]

  [cyan]/help[/cyan]       Show this help message
  [cyan]/clear[/cyan]      Clear conversation history (start fresh)
  [cyan]/history[/cyan]    Show number of messages in current session
  [cyan]/quit[/cyan]       Exit the chatbot

[bold]Tips[/bold]
  • Ask about any financial topic from earnings calls
  • Use [cyan]/clear[/cyan] to reset context between different companies
"""


# ---------------------------------------------------------------------------
# Session commands handler
# ---------------------------------------------------------------------------

def handle_command(command: str, history: ConversationHistory) -> bool:
    """
    Process slash commands.
    Returns True if the main loop should continue, False if it should exit.
    """
    cmd = command.strip().lower()

    if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
        console.print("\n[dim]Goodbye! 👋[/dim]\n")
        return False

    elif cmd == "/help":
        console.print(Panel(HELP_TEXT, border_style="dim", padding=(1, 2)))

    elif cmd == "/clear":
        history.clear()
        console.print("[dim]✓ Conversation history cleared.[/dim]\n")

    elif cmd == "/history":
        turn_count = len(history) // 2
        console.print(
            f"[dim]Session has [bold]{len(history)}[/bold] messages "
            f"({turn_count} turns).[/dim]\n"
        )

    else:
        console.print(f"[yellow]Unknown command:[/yellow] {command}. Type [cyan]/help[/cyan] for options.\n")

    return True


# ---------------------------------------------------------------------------
# Response rendering
# ---------------------------------------------------------------------------

def stream_response(client: LLMClient, history: ConversationHistory) -> str:
    """Stream response to terminal, return full text when done."""
    full_response = ""
    console.print()
    console.print("[bold cyan]Assistant:[/bold cyan] ", end="")

    try:
        for chunk in client.chat_stream(history):
            console.print(chunk, end="", markup=False)
            full_response += chunk
        console.print("\n")
    except RuntimeError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        return ""

    return full_response


def print_response(client: LLMClient, history: ConversationHistory) -> str:
    """Non-streaming: show spinner, then render as markdown."""
    from rich.spinner import Spinner
    from rich.live import Live

    full_response = ""

    with Live(
        Spinner("dots", text=" Thinking...", style="dim"),
        console=console,
        transient=True,
    ):
        try:
            response = client.chat(history)
            full_response = response.content
        except RuntimeError as e:
            console.print(f"\n[red]Error:[/red] {e}\n")
            return ""

    console.print()
    console.print("[bold cyan]Assistant:[/bold cyan]")
    console.print(Markdown(full_response))
    console.print(
        f"[dim]  tokens: {response.prompt_tokens}↑ {response.completion_tokens}↓[/dim]\n"
    )
    return full_response


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

@app.command()
def chat(
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response tokens"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override LLM model"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug info"),
):
    """
    Start an interactive financial Q&A chat session.
    """
    # Apply CLI overrides
    if model:
        settings.model = model

    # Validate environment early
    try:
        settings.validate()
    except EnvironmentError as e:
        console.print(f"[red bold]Configuration Error:[/red bold] {e}")
        console.print("[dim]Create a .env file with OPENAI_API_KEY=your_key_here[/dim]")
        raise typer.Exit(code=1)

    # Init
    client = LLMClient()
    history = ConversationHistory()

    # Welcome
    console.print(Panel(
        WELCOME_BANNER.format(model=client.model),
        border_style="cyan",
        padding=(0, 2),
    ))

    if verbose:
        console.print(f"[dim]Model: {client.model} | Stream: {stream} | Temp: {settings.temperature}[/dim]\n")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    while True:
        try:
            # Prompt
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Interrupted. Goodbye! 👋[/dim]\n")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            should_continue = handle_command(user_input, history)
            if not should_continue:
                break
            continue

        # Add to history
        history.add_user(user_input)

        # Get + display response
        if stream:
            response_text = stream_response(client, history)
        else:
            response_text = print_response(client, history)

        # Save assistant reply to history (skip on error)
        if response_text:
            history.add_assistant(response_text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app()


if __name__ == "__main__":
    main()