"""
Financial Q&A CLI — Part 1 + Part 2

Commands:
    python main.py chat                          → RAG chat (uses loaded transcripts)
    python main.py ingest ./data/transcripts/    → ingest a folder of PDFs
    python main.py ingest report.pdf             → ingest a single PDF
    python main.py list                          → show indexed transcripts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from core.llm import ConversationHistory
from core.rag import RAGEngine
from core.vectorstore import VectorStore, SearchFilters
from core.config import settings

app = typer.Typer(
    name="financial-qa",
    help="Financial Earnings Call Q&A Assistant",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# COMMAND: ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to a PDF file or directory of PDFs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ingest earnings call transcript PDF(s) into the vector store."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        settings.validate()
    except EnvironmentError as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(1)

    from ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    target = Path(path)
    console.print()

    if target.is_dir():
        console.print(f"[bold]Ingesting directory:[/bold] {target}\n")
        batch = pipeline.ingest_directory(target)

        table = Table(title="Ingestion Summary", show_lines=True)
        table.add_column("File", style="dim")
        table.add_column("Company")
        table.add_column("Quarter")
        table.add_column("FY")
        table.add_column("Chunks", justify="right")
        table.add_column("Status")

        for r in batch.results:
            status = "[green]✓[/green]" if r.success else f"[red]✗ {r.error[:40]}[/red]"
            table.add_row(
                r.source_file[:40], r.company, r.quarter, r.fiscal_year,
                str(r.chunks_added) if r.success else "-", status,
            )

        console.print(table)
        console.print(
            f"\n[green]Done.[/green] {batch.successful}/{batch.total_files} files, "
            f"{batch.total_chunks} chunks stored.\n"
        )

    elif target.is_file() and target.suffix.lower() == ".pdf":
        console.print(f"[bold]Ingesting:[/bold] {target.name}\n")
        with console.status("[dim]Parsing, chunking, embedding...[/dim]"):
            result = pipeline.ingest_file(target)
        if result.success:
            console.print(
                f"[green]✓[/green] {result.company} {result.quarter} {result.fiscal_year} "
                f"— {result.chunks_added} chunks stored.\n"
            )
        else:
            console.print(f"[red]✗ Failed:[/red] {result.error}\n")
            raise typer.Exit(1)
    else:
        console.print(f"[red]Error:[/red] '{path}' is not a PDF file or directory.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# COMMAND: list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_docs():
    """List all transcripts currently indexed in the vector store."""
    try:
        store = VectorStore()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    docs = store.list_documents()
    console.print()

    if not docs:
        console.print("[dim]No transcripts indexed yet. Run:[/dim]")
        console.print("  [cyan]python main.py ingest ./data/transcripts/[/cyan]\n")
        return

    table = Table(title=f"Indexed Transcripts ({len(docs)})", show_lines=True)
    table.add_column("Company", style="bold")
    table.add_column("Quarter")
    table.add_column("Fiscal Year")
    table.add_column("Date")
    table.add_column("Source File", style="dim")

    for doc in docs:
        table.add_row(doc["company"], doc["quarter"], doc["fiscal_year"], doc["date"], doc["source_file"])

    console.print(table)
    console.print(f"\n[dim]Total chunks in store: {store.count()}[/dim]\n")


# ---------------------------------------------------------------------------
# COMMAND: chat
# ---------------------------------------------------------------------------

@app.command()
def chat(
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Filter by company name"),
    quarter: Optional[str] = typer.Option(None, "--quarter", "-q", help="Filter by quarter e.g. Q1"),
    fiscal_year: Optional[str] = typer.Option(None, "--fy", help="Filter by fiscal year e.g. FY25"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override LLM model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start an interactive RAG-powered Q&A session over earnings transcripts."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    if model:
        settings.model = model

    try:
        settings.validate()
    except EnvironmentError as e:
        console.print(f"[red]Config error:[/red] {e}\n[dim]Add OPENAI_API_KEY to .env[/dim]")
        raise typer.Exit(1)

    filters = SearchFilters(company=company, quarter=quarter, fiscal_year=fiscal_year) \
        if any([company, quarter, fiscal_year]) else None

    try:
        engine = RAGEngine()
    except Exception as e:
        console.print(f"[red]Startup error:[/red] {e}")
        raise typer.Exit(1)

    doc_count = engine.store.count()
    if doc_count == 0:
        console.print(Panel(
            "[yellow]No transcripts indexed yet.[/yellow]\n\n"
            "Run first:\n  [cyan]python main.py ingest ./data/transcripts/[/cyan]",
            border_style="yellow",
        ))
        raise typer.Exit(1)

    # Build filter display
    filter_parts = []
    if company:     filter_parts.append(f"company=[cyan]{company}[/cyan]")
    if quarter:     filter_parts.append(f"quarter=[cyan]{quarter}[/cyan]")
    if fiscal_year: filter_parts.append(f"fy=[cyan]{fiscal_year}[/cyan]")
    filter_info = f"\n[dim]Filters: {', '.join(filter_parts)}[/dim]" if filter_parts else ""

    console.print(Panel(
        f"[bold cyan]Financial Q&A Assistant[/bold cyan]\n"
        f"[dim]{doc_count} chunks indexed · {settings.model}[/dim]{filter_info}\n\n"
        f"[dim]Type [cyan]/sources[/cyan] after any answer · [cyan]/clear[/cyan] · [cyan]/quit[/cyan][/dim]",
        border_style="cyan", padding=(0, 2),
    ))

    history = ConversationHistory()
    last_sources = []

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 👋[/dim]\n")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower().strip()
            if cmd in ("/quit", "/exit", "/q"):
                console.print("\n[dim]Goodbye! 👋[/dim]\n")
                break
            elif cmd == "/clear":
                history.clear()
                console.print("[dim]✓ Conversation cleared.[/dim]")
            elif cmd == "/sources":
                if not last_sources:
                    console.print("[dim]No sources yet.[/dim]")
                else:
                    console.print()
                    for i, src in enumerate(last_sources, 1):
                        m = src.metadata
                        console.print(
                            f"[{i}] [bold]{m.get('company')}[/bold] "
                            f"{m.get('quarter')} {m.get('fiscal_year')} | "
                            f"{m.get('section')} | score: [dim]{src.score:.3f}[/dim]"
                        )
                        console.print(f"    [dim]{src.text[:150]}...[/dim]\n")
            elif cmd == "/help":
                console.print(Panel(
                    "[cyan]/sources[/cyan]  Show sources from last answer\n"
                    "[cyan]/clear[/cyan]    Reset conversation history\n"
                    "[cyan]/quit[/cyan]     Exit\n\n"
                    "[bold]Startup filters:[/bold]\n"
                    "  python main.py chat --company Birlasoft --quarter Q1",
                    border_style="dim",
                ))
            else:
                console.print(f"[yellow]Unknown command:[/yellow] {user_input}")
            continue

        # RAG query
        with console.status("[dim]Searching transcripts...[/dim]"):
            try:
                response = engine.query(user_input, history, filters)
            except RuntimeError as e:
                console.print(f"[red]Error:[/red] {e}")
                continue

        last_sources = response.sources
        console.print()
        console.print("[bold cyan]Assistant:[/bold cyan]")
        console.print(Markdown(response.answer))

        if response.used_rag and response.sources:
            companies = list({s.metadata.get("company", "?") for s in response.sources})
            quarters  = list({f"{s.metadata.get('quarter')} {s.metadata.get('fiscal_year')}" for s in response.sources})
            console.print(
                f"\n[dim]  📎 {len(response.sources)} source(s) · "
                f"{', '.join(companies)} · {', '.join(quarters)} · /sources for details[/dim]"
            )


def main():
    app()


if __name__ == "__main__":
    main()