"""vTTS Command Line Interface"""

import click
import subprocess
import sys
from rich.console import Console
from rich.table import Table
from loguru import logger

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


@main.command()
@click.option("--fix", is_flag=True, help="문제를 자동으로 수정합니다")
@click.option("--cuda", is_flag=True, help="CUDA 지원을 설치합니다")
def doctor(fix: bool, cuda: bool):
    """환경을 진단하고 문제를 수정합니다.
    
    Examples:
        vtts doctor          # 환경 진단
        vtts doctor --fix    # 자동 수정
        vtts doctor --cuda   # CUDA 지원 설치
    """
    import torch
    
    console.print("\n[bold]🩺 vTTS Environment Diagnosis[/bold]\n")
    
    issues = []
    
    # ============================================================
    # 1. Python 버전 확인
    # ============================================================
    py_version = sys.version_info
    py_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
    
    if py_version >= (3, 10) and py_version < (3, 13):
        console.print(f"[green]✓[/green] Python: {py_str}")
    else:
        console.print(f"[red]✗[/red] Python: {py_str} (3.10-3.12 권장)")
        issues.append("python")
    
    # ============================================================
    # 2. numpy 버전 확인
    # ============================================================
    try:
        import numpy as np
        np_version = np.__version__
        
        # numpy 2.0 이상은 호환성 문제 있음
        major = int(np_version.split('.')[0])
        if major >= 2:
            console.print(f"[red]✗[/red] numpy: {np_version} (1.24-1.26 권장, 2.x 호환성 문제)")
            issues.append("numpy")
        else:
            console.print(f"[green]✓[/green] numpy: {np_version}")
    except ImportError:
        console.print("[red]✗[/red] numpy: 설치되지 않음")
        issues.append("numpy")
    
    # ============================================================
    # 3. ONNX Runtime 확인
    # ============================================================
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        providers = ort.get_available_providers()
        
        has_cuda = "CUDAExecutionProvider" in providers
        
        if has_cuda:
            console.print(f"[green]✓[/green] onnxruntime: {ort_version} (CUDA 지원)")
        else:
            console.print(f"[yellow]![/yellow] onnxruntime: {ort_version} (CPU 전용)")
            if cuda or torch.cuda.is_available():
                issues.append("onnxruntime-gpu")
        
        console.print(f"  [dim]Providers: {', '.join(providers)}[/dim]")
        
    except ImportError:
        console.print("[red]✗[/red] onnxruntime: 설치되지 않음")
        issues.append("onnxruntime")
    
    # ============================================================
    # 4. PyTorch & CUDA 확인
    # ============================================================
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[green]✓[/green] PyTorch: {torch_version} (CUDA {cuda_version})")
        console.print(f"  [dim]GPU: {gpu_name}[/dim]")
    else:
        console.print(f"[yellow]![/yellow] PyTorch: {torch_version} (CPU 전용)")
    
    # ============================================================
    # 5. vTTS 확인
    # ============================================================
    try:
        import vtts
        console.print(f"[green]✓[/green] vTTS: 설치됨")
    except ImportError:
        console.print("[red]✗[/red] vTTS: 설치되지 않음")
        issues.append("vtts")
    
    # ============================================================
    # 결과 요약
    # ============================================================
    console.print()
    
    if not issues:
        console.print("[bold green]✅ 모든 환경이 정상입니다![/bold green]\n")
        return
    
    console.print(f"[bold yellow]⚠️ {len(issues)}개의 문제가 발견되었습니다:[/bold yellow]")
    for issue in issues:
        console.print(f"  - {issue}")
    
    if not fix:
        console.print("\n[dim]자동 수정: vtts doctor --fix[/dim]")
        console.print("[dim]CUDA 설치: vtts doctor --fix --cuda[/dim]\n")
        return
    
    # ============================================================
    # 자동 수정
    # ============================================================
    console.print("\n[bold]🔧 자동 수정 중...[/bold]\n")
    
    # numpy 수정
    if "numpy" in issues:
        console.print("[cyan]→[/cyan] numpy 재설치 중...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y", "-q"], 
                      capture_output=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=1.24.0,<2.0.0", "-q"],
                      capture_output=True)
        console.print("[green]✓[/green] numpy 설치 완료")
    
    # onnxruntime 수정
    if "onnxruntime" in issues or "onnxruntime-gpu" in issues:
        console.print("[cyan]→[/cyan] onnxruntime 재설치 중...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "onnxruntime-gpu", "-y", "-q"],
                      capture_output=True)
        
        if cuda or torch.cuda.is_available():
            subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu>=1.16.0", "-q"],
                          capture_output=True)
            console.print("[green]✓[/green] onnxruntime-gpu 설치 완료")
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime>=1.16.0", "-q"],
                          capture_output=True)
            console.print("[green]✓[/green] onnxruntime 설치 완료")
    
    console.print("\n[bold green]✅ 수정 완료![/bold green]")
    console.print("[dim]변경사항 적용을 위해 Python을 재시작하세요.[/dim]\n")


@main.command()
@click.option("--engine", default="supertonic", help="설치할 엔진 (supertonic, gptsovits, cosyvoice, all)")
@click.option("--cuda/--no-cuda", default=True, help="CUDA 지원 여부")
def setup(engine: str, cuda: bool):
    """엔진별 의존성을 설치합니다.
    
    Examples:
        vtts setup --engine supertonic         # Supertonic (CPU)
        vtts setup --engine supertonic --cuda  # Supertonic (GPU)
        vtts setup --engine gptsovits          # GPT-SoVITS (저장소 자동 클론)
        vtts setup --engine all                # 모든 엔진
    """
    import torch
    import os
    from pathlib import Path
    
    console.print(f"\n[bold]📦 vTTS 엔진 설치: {engine}[/bold]\n")
    
    # CUDA 자동 감지
    if cuda and not torch.cuda.is_available():
        console.print("[yellow]⚠️ CUDA가 감지되지 않았습니다. CPU 모드로 설치합니다.[/yellow]")
        cuda = False
    
    total_steps = 4 if engine in ["gptsovits", "all"] else 3
    step = 1
    
    # numpy 먼저 설치 (호환성)
    console.print(f"[cyan]→[/cyan] [{step}/{total_steps}] numpy 호환성 확인...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y", "-q"],
                  capture_output=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=1.24.0,<2.0.0", "-q"],
                  capture_output=True)
    step += 1
    
    # onnxruntime 설치
    console.print(f"[cyan]→[/cyan] [{step}/{total_steps}] onnxruntime 설치...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "onnxruntime-gpu", "-y", "-q"],
                  capture_output=True)
    
    if engine in ["supertonic", "all"] and cuda:
        subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu>=1.16.0", "-q"],
                      capture_output=True)
    elif engine == "supertonic":
        subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime>=1.16.0", "-q"],
                      capture_output=True)
    step += 1
    
    # ============================================================
    # GPT-SoVITS: 저장소 자동 클론 및 설치
    # ============================================================
    if engine in ["gptsovits", "all"]:
        console.print(f"[cyan]→[/cyan] [{step}/{total_steps}] GPT-SoVITS 저장소 설치...")
        
        # 설치 경로 결정
        gpt_sovits_path = os.environ.get("GPT_SOVITS_PATH")
        
        if not gpt_sovits_path:
            # 기본 경로: ~/.vtts/GPT-SoVITS
            vtts_dir = Path.home() / ".vtts"
            vtts_dir.mkdir(exist_ok=True)
            gpt_sovits_path = vtts_dir / "GPT-SoVITS"
        else:
            gpt_sovits_path = Path(gpt_sovits_path)
        
        if gpt_sovits_path.exists():
            console.print(f"  [dim]GPT-SoVITS already exists: {gpt_sovits_path}[/dim]")
            console.print("  [dim]Pulling latest changes...[/dim]")
            result = subprocess.run(
                ["git", "-C", str(gpt_sovits_path), "pull"],
                capture_output=True, text=True
            )
        else:
            console.print(f"  [dim]Cloning to: {gpt_sovits_path}[/dim]")
            result = subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/RVC-Boss/GPT-SoVITS.git",
                 str(gpt_sovits_path)],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                console.print(f"[red]❌ Git clone failed: {result.stderr}[/red]")
                return
        
        # GPT-SoVITS requirements 설치
        console.print("  [dim]Installing GPT-SoVITS requirements...[/dim]")
        req_file = gpt_sovits_path / "requirements.txt"
        
        if req_file.exists():
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                console.print(f"[yellow]⚠️ Some requirements failed, but continuing...[/yellow]")
        
        # 환경변수 설정 안내
        console.print(f"\n[green]✓[/green] GPT-SoVITS installed: {gpt_sovits_path}")
        
        # 자동으로 환경변수 설정 (현재 세션)
        os.environ["GPT_SOVITS_PATH"] = str(gpt_sovits_path)
        
        console.print("\n[bold yellow]⚠️ 환경변수를 영구 설정하려면:[/bold yellow]")
        console.print(f"  [dim]export GPT_SOVITS_PATH={gpt_sovits_path}[/dim]")
        console.print(f"  [dim]위 명령을 ~/.bashrc 또는 ~/.zshrc에 추가하세요[/dim]")
        
        step += 1
    
    # 엔진별 의존성 설치
    console.print(f"[cyan]→[/cyan] [{step}/{total_steps}] {engine} 의존성 설치...")
    
    extras = {
        "supertonic": "supertonic-cuda" if cuda else "supertonic",
        "gptsovits": "gptsovits",
        "cosyvoice": "cosyvoice",
        "all": "all"
    }
    
    extra = extras.get(engine, "supertonic")
    
    # GitHub에서 설치
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         f"vtts[{extra}] @ git+https://github.com/bellkjtt/vTTS.git"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        console.print(f"\n[bold green]✅ {engine} 엔진 설치 완료![/bold green]")
        
        if engine == "gptsovits":
            console.print("\n[dim]사용법: vtts serve kevinwang676/GPT-SoVITS-v3 --device cuda[/dim]")
            console.print("[dim]참고: reference_audio와 reference_text 파라미터가 필수입니다![/dim]\n")
        else:
            console.print("\n[dim]사용법: vtts serve Supertone/supertonic-2[/dim]\n")
    else:
        console.print(f"\n[bold red]❌ 설치 실패[/bold red]")
        console.print(f"[dim]{result.stderr}[/dim]")


if __name__ == "__main__":
    main()
