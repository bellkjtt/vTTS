"""vTTS Command Line Interface"""

import click
from rich.console import Console
from rich.table import Table
from loguru import logger
import sys

from vtts.engines.registry import EngineRegistry
from vtts.server.app import create_app

console = Console()


@click.group()
@click.version_option()
def main():
    """vTTS - Universal TTS Serving System
    
    vLLM for Text-to-Speech
    """
    pass


@main.command()
@click.argument("model_id")
@click.option("--host", default="0.0.0.0", help="서버 호스트")
@click.option("--port", default=8000, help="서버 포트")
@click.option("--device", default="auto", help="디바이스 (cuda, cpu, auto)")
@click.option("--workers", default=1, help="워커 수")
@click.option("--cache-dir", default=None, help="모델 캐시 디렉토리")
@click.option("--log-level", default="INFO", help="로그 레벨")
@click.option("--stt-model", default=None, help="STT 모델 (Whisper, 선택적)")
def serve(model_id: str, host: str, port: int, device: str, workers: int, cache_dir: str, log_level: str, stt_model: str):
    """TTS 모델 서버를 시작합니다.
    
    Examples:
        vtts serve Supertone/supertonic-2
        vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000
        vtts serve kevinwang676/GPT-SoVITS-v3 --device cuda:0
    """
    # 로그 설정
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    console.print(f"[bold green]🚀 Starting vTTS Server[/bold green]")
    console.print(f"Model: [cyan]{model_id}[/cyan]")
    console.print(f"Host: [cyan]{host}:{port}[/cyan]")
    console.print(f"Device: [cyan]{device}[/cyan]")
    
    # 엔진 확인
    engine_class = EngineRegistry.get_engine_for_model(model_id)
    if engine_class is None:
        console.print(f"[bold red]❌ No engine found for model: {model_id}[/bold red]")
        console.print("\n사용 가능한 엔진:")
        list_models()
        sys.exit(1)
    
    console.print(f"Engine: [cyan]{engine_class.__name__}[/cyan]")
    
    # STT 모델 확인
    if stt_model:
        console.print(f"STT Model: [cyan]{stt_model}[/cyan]")
    
    # 서버 실행
    import uvicorn
    
    app = create_app(
        model_id=model_id,
        device=device,
        cache_dir=cache_dir,
        stt_model_id=stt_model
    )
    
    console.print("\n[bold green]✓ Server starting...[/bold green]")
    console.print(f"[dim]OpenAI compatible API: http://{host}:{port}/v1[/dim]")
    console.print(f"[dim]Docs: http://{host}:{port}/docs[/dim]\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level.lower()
    )


@main.command()
def list_models():
    """지원하는 모델 목록을 표시합니다."""
    console.print("\n[bold]지원하는 TTS 엔진 및 모델:[/bold]\n")
    
    supported = EngineRegistry.list_supported_models()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("엔진", style="cyan", width=15)
    table.add_column("지원 모델 패턴", style="green")
    table.add_column("예시", style="yellow")
    
    examples = {
        "supertonic": "Supertone/supertonic-2",
        "cosyvoice": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "gptsovits": "kevinwang676/GPT-SoVITS-v3"
    }
    
    for engine_name, patterns in supported.items():
        table.add_row(
            engine_name,
            ", ".join(patterns),
            examples.get(engine_name, "-")
        )
    
    console.print(table)
    console.print()


@main.command()
@click.argument("model_id")
def info(model_id: str):
    """모델 정보를 표시합니다."""
    console.print(f"\n[bold]Model Information:[/bold] [cyan]{model_id}[/cyan]\n")
    
    engine_class = EngineRegistry.get_engine_for_model(model_id)
    
    if engine_class is None:
        console.print(f"[bold red]❌ No engine found for model: {model_id}[/bold red]")
        sys.exit(1)
    
    # 임시로 엔진 인스턴스 생성 (모델 로드하지 않음)
    try:
        engine = engine_class(model_id=model_id)
        info_dict = engine.get_model_info()
        
        table = Table(show_header=False)
        table.add_column("속성", style="cyan", width=25)
        table.add_column("값", style="green")
        
        table.add_row("Engine", engine_class.__name__)
        table.add_row("Model ID", info_dict["model_id"])
        table.add_row("Device", info_dict["device"])
        table.add_row("Sample Rate", f"{info_dict['sample_rate']} Hz")
        table.add_row("Streaming Support", "✓" if info_dict["supports_streaming"] else "✗")
        table.add_row("Zero-shot Support", "✓" if info_dict["supports_zero_shot"] else "✗")
        table.add_row("Supported Languages", ", ".join(info_dict["supported_languages"]))
        
        console.print(table)
        console.print()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
