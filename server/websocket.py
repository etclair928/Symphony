# =================================================================
# MODULE: server/websocket.py
# DESCRIPTION: WebSocket + HTTP server for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-12
#
# CRITICAL FIXES APPLIED:
#   1. WebSocket connection tracking for real-time progress
#   2. asyncio.run_coroutine_threadsafe for cross-thread messaging
#   3. Session reconnection support (browser refresh resilience)
#   4. Progress amplification from pipeline to client
#   5. Forensic progress markers for UI highlighting
#
# Authored by: DeepSeek - Complete 5.0 rewrite with WebSocket fixes (2026-05-12)
# =================================================================

import asyncio
import json
import base64
import hashlib
import numpy as np
import time
import uuid
import tempfile
import shutil
import logging
import threading
from typing import Dict, Set, Optional, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add parent directory for imports
import sys

# Force UTF-8 stdout/stderr so Unicode status characters used elsewhere
# in this codebase don't crash on Windows consoles (default cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from core.order_types import (
    NoteEvent, AudioContext, StageResult, Confidence
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    MEMORY_LIMIT_MB,
    STAGGERED_GC_TRIGGER_MB
)
from core.guided_control import GuidedParams

# Orchestration imports
from orchestration.pipeline import GrimlockPipeline, PipelineResult
from orchestration.music_box import MusicBox, MusicBoxConfig, LogLevel, DecisionType
from orchestration.scribe import Scribe

# Memory management
from memory.guardian import MemoryGuardian

# Agent factory
from agents.factory import AgentFactory

# Audio loading
from ingestion.loader import AudioLoader

# WebSocket libraries
try:
    import websockets
    from websockets.server import WebSocketServerProtocol, serve

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Audio processing
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf

    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grimlock_server")


# ========================================================================
# Enums and Types
# ========================================================================

class MessageType(str, Enum):
    """WebSocket message types."""
    # Client -> Server
    AUDIO_CHUNK = "audio_chunk"
    AUDIO_START = "audio_start"
    AUDIO_END = "audio_end"
    PING = "ping"
    CONFIGURE = "configure"
    RESUME_JOB = "resume_job"  # NEW: Resume progress for existing job

    # Server -> Client
    TRANSCRIPTION_CHUNK = "transcription_chunk"
    TRANSCRIPTION_FINAL = "transcription_final"
    STATUS = "status"
    ERROR = "error"
    PONG = "pong"
    READY = "ready"
    JOB_UPDATE = "job_update"
    UPLOAD_ACK = "upload_ack"
    LOG_ENTRY = "log_entry"
    PROGRESS = "progress"  # NEW: Explicit progress message
    STAGE_COMPLETE = "stage_complete"  # NEW: Stage completion notification
    FORENSIC_MARKER = "forensic_marker"  # NEW: Highlight difficult sections


class JobStatus(str, Enum):
    """Job status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ========================================================================
# Data Classes
# ========================================================================

@dataclass
class TranscriptionJob:
    """Background transcription job state."""
    job_id: str
    audio_path: str
    original_filename: str
    status: JobStatus = JobStatus.PENDING
    stage: str = "queued"
    progress: int = 0
    notes: List[Dict] = field(default_factory=list)
    veto_count: int = 0
    vetoes: List[Dict] = field(default_factory=list)
    forensic_markers: List[Dict] = field(default_factory=list)  # NEW: Problem sections
    avg_confidence: float = 0.0
    elapsed_sec: float = 0.0
    error: Optional[str] = None
    result: Optional[PipelineResult] = None
    guided_params: Optional[Dict] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    midi_path: Optional[str] = None
    json_path: Optional[str] = None
    music_box_session: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to serializable dict for API responses."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "stage": self.stage,
            "progress": self.progress,
            "notes": self.notes[:100],
            "veto_count": self.veto_count,
            "forensic_markers": self.forensic_markers[:20],  # NEW
            "avg_confidence": self.avg_confidence,
            "elapsed_sec": self.elapsed_sec,
            "error": self.error,
            "original_filename": self.original_filename,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class ClientSession:
    """WebSocket client session state."""
    session_id: str
    websocket: Any  # Store the actual WebSocket connection
    connected_at: datetime
    last_activity: datetime
    audio_buffer: List[np.ndarray] = field(default_factory=list)
    total_samples: int = 0
    pipeline: Optional[GrimlockPipeline] = None
    music_box: Optional[MusicBox] = None
    scribe: Optional[Scribe] = None
    guardian: Optional[MemoryGuardian] = None
    config: Dict[str, Any] = field(default_factory=dict)
    current_job_id: Optional[str] = None
    chunk_count: int = 0
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add a chunk of audio data to the buffer."""
        self.audio_buffer.append(audio_data)
        self.total_samples += len(audio_data)
        self.update_activity()

    def get_full_audio(self) -> np.ndarray:
        """Get the complete audio buffer."""
        if not self.audio_buffer:
            return np.array([])
        return np.concatenate(self.audio_buffer)

    def clear_audio(self):
        """Clear the audio buffer."""
        self.audio_buffer.clear()
        self.total_samples = 0

    def get_duration_seconds(self, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
        """Get the total duration of buffered audio in seconds."""
        return self.total_samples / sample_rate

    async def send_json(self, data: Dict):
        """Thread-safe send_json with connection check."""
        async with self._send_lock:
            try:
                if self.websocket and not getattr(self.websocket, 'closed', False):
                    await self.websocket.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to send to session {self.session_id}: {e}")


# ========================================================================
# Progress Callback (with WebSocket integration)
# ========================================================================

class JobProgressCallback:
    """
    Callback for updating job progress during transcription.

    CRITICAL FIX: Uses asyncio.run_coroutine_threadsafe to send
    messages from background threads back to the WebSocket event loop.
    """

    def __init__(self, job: TranscriptionJob, websocket_server: 'WebSocketServer', loop: asyncio.AbstractEventLoop):
        self.job = job
        self.server = websocket_server
        self.loop = loop
        self._last_update = time.time()
        self._last_stage = ""

    def __call__(self, stage: str, progress: int, notes: List[NoteEvent] = None):
        """Update job progress - called from background thread."""
        self.job.stage = stage
        self.job.progress = min(100, max(0, progress))

        if notes:
            self.job.notes = [self._note_to_dict(n) for n in notes[:500]]
            if notes:
                self.job.avg_confidence = sum(n.confidence for n in notes) / len(notes)

        # Throttle updates to avoid overwhelming
        now = time.time()
        stage_changed = stage != self._last_stage
        if now - self._last_update > 0.5 or progress == 100 or stage_changed:
            self._last_update = now
            self._last_stage = stage

            # Send progress update via WebSocket (CRITICAL: cross-thread)
            if self.job.job_id in self.server._job_sessions:
                session_id = self.server._job_sessions[self.job.job_id]
                if session_id in self.server._sessions:
                    session = self.server._sessions[session_id]

                    # Create progress message
                    message = {
                        "type": MessageType.PROGRESS.value,
                        "job_id": self.job.job_id,
                        "stage": stage,
                        "progress": progress,
                        "note_count": len(notes) if notes else 0,
                        "confidence": self.job.avg_confidence,
                        "veto_count": self.job.veto_count
                    }

                    # Send from background thread to asyncio event loop
                    if self.loop and not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            session.send_json(message),
                            self.loop
                        )

                    # Also send stage completion marker
                    if stage_changed and progress > 0 and progress < 100:
                        stage_msg = {
                            "type": MessageType.STAGE_COMPLETE.value,
                            "job_id": self.job.job_id,
                            "completed_stage": stage,
                            "timestamp": time.time()
                        }
                        asyncio.run_coroutine_threadsafe(
                            session.send_json(stage_msg),
                            self.loop
                        )

    def add_forensic_marker(self, stage: str, time_sec: float, reason: str, confidence: float):
        """
        Add a forensic marker for problematic sections.

        This allows the UI to highlight difficult bars in real-time.
        """
        marker = {
            "stage": stage,
            "time_sec": time_sec,
            "reason": reason,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        self.job.forensic_markers.append(marker)

        # Send immediately to client
        if self.job.job_id in self.server._job_sessions:
            session_id = self.server._job_sessions[self.job.job_id]
            if session_id in self.server._sessions:
                session = self.server._sessions[session_id]
                message = {
                    "type": MessageType.FORENSIC_MARKER.value,
                    "job_id": self.job.job_id,
                    "marker": marker
                }
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        session.send_json(message),
                        self.loop
                    )

    @staticmethod
    def _note_to_dict(note: NoteEvent) -> Dict:
        """Convert NoteEvent to serializable dict."""
        return {
            "pitch": note.pitch,
            "pitch_name": _midi_to_name(note.pitch),
            "start_ms": note.start_ms,
            "end_ms": note.end_ms,
            "velocity": note.velocity,
            "confidence": note.confidence,
            "source": note.source.value if hasattr(note.source, 'value') else str(note.source),
            "is_snapped": note.is_snapped() if hasattr(note, 'is_snapped') else False
        }


def _midi_to_name(midi: int) -> str:
    """Convert MIDI note number to note name."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi // 12 - 1
    return f"{notes[midi % 12]}{octave}"


# ========================================================================
# WebSocket Server (FULLY WIRED)
# ========================================================================

class WebSocketServer:
    """
    WebSocket + HTTP server for Grimlock 5.0 transcription.

    FIXED: Now tracks WebSocket connections and sends real-time progress.
    """

    def __init__(
            self,
            host: str = "localhost",
            port: int = 8000,
            use_fastapi: bool = True,
            max_concurrent_jobs: int = 4,
            job_timeout_seconds: int = 3600
    ):
        """
        Args:
            host: Server host address
            port: Server port
            use_fastapi: Use FastAPI (with HTTP endpoints) or pure WebSockets
            max_concurrent_jobs: Maximum number of concurrent transcription jobs
            job_timeout_seconds: Timeout for jobs in seconds
        """
        self.host = host
        self.port = port
        self.use_fastapi = use_fastapi and FASTAPI_AVAILABLE
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_timeout_seconds = job_timeout_seconds

        # Session management (FIXED: now stores WebSocket connections)
        self._sessions: Dict[str, ClientSession] = {}
        self._jobs: Dict[str, TranscriptionJob] = {}
        self._job_sessions: Dict[str, str] = {}  # job_id -> session_id
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self._running = False
        self._start_time: Optional[float] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Directories
        self._temp_dir = Path(tempfile.gettempdir()) / "grimlock_jobs"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = Path("output")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"WebSocketServer initialized on {host}:{port}")

    async def start(self):
        """Start WebSocket + HTTP server."""
        self._running = True
        self._start_time = time.time()
        self._event_loop = asyncio.get_running_loop()

        if self.use_fastapi:
            await self._start_fastapi()
        else:
            await self._start_websockets()

        logger.info(f"Grimlock 5.0 server started on http://{self.host}:{self.port}")

    async def _start_fastapi(self):
        """Start server using FastAPI with WebSocket + HTTP endpoints."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")

        app = FastAPI(
            title="Grimlock 5.0 Transcription Server",
            description="Classic transcription workstation with guided mode",
            version="5.0.0"
        )

        # Enable CORS for frontend
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # =========================================================
        # HTTP Endpoints
        # =========================================================

        @app.get("/")
        async def get_index():
            """Serve the classic interface HTML."""
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                return HTMLResponse(html_path.read_text())
            return HTMLResponse(self._get_classic_interface_html())

        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return JSONResponse({
                "status": "healthy",
                "version": "5.0.0",
                "uptime_seconds": time.time() - (self._start_time or time.time()),
                "active_jobs": len([j for j in self._jobs.values() if j.status == JobStatus.RUNNING]),
                "total_jobs": len(self._jobs),
                "active_sessions": len(self._sessions)
            })

        @app.post("/upload")
        async def upload_audio(
                background_tasks: BackgroundTasks,
                audio: UploadFile = File(...),
                guided: str = Form(None)
        ):
            """Upload audio file for transcription. Supports guided mode parameters."""
            job_id = str(uuid.uuid4())[:8]

            if not audio.filename:
                return JSONResponse({"error": "No file provided"}, status_code=400)

            sanitized_filename = audio.filename.replace('/', '_').replace('\\', '_')
            safe_filename = f"{job_id}_{sanitized_filename}"
            temp_path = self._temp_dir / safe_filename

            content = await audio.read()
            temp_path.write_bytes(content)

            try:
                loader = AudioLoader()
                info = loader.get_info(temp_path)
                if not info.is_supported:
                    return JSONResponse({"error": f"Unsupported format: {temp_path.suffix}"}, status_code=400)
            except Exception as e:
                return JSONResponse({"error": f"Invalid audio file: {e}"}, status_code=400)

            guided_params = None
            if guided:
                try:
                    guided_params = json.loads(guided)
                except json.JSONDecodeError:
                    guided_params = {"error": "invalid json"}

            job = TranscriptionJob(
                job_id=job_id,
                audio_path=str(temp_path),
                original_filename=audio.filename,
                guided_params=guided_params,
                status=JobStatus.PENDING
            )
            self._jobs[job_id] = job

            background_tasks.add_task(self._run_transcription, job_id)

            logger.info(f"Job {job_id} created for {audio.filename}")

            return JSONResponse({
                "job_id": job_id,
                "status": "processing",
                "message": "Transcription started"
            })

        @app.get("/status/{job_id}")
        async def get_status(job_id: str):
            """Get transcription job status."""
            job = self._jobs.get(job_id)
            if not job:
                return JSONResponse({"error": "Job not found"}, status_code=404)
            return JSONResponse(job.to_dict())

        @app.get("/download/{job_id}/midi")
        async def download_midi(job_id: str):
            """Download MIDI file for completed job."""
            job = self._jobs.get(job_id)
            if not job:
                return JSONResponse({"error": "Job not found"}, status_code=404)
            if job.status != JobStatus.COMPLETED or not job.midi_path:
                return JSONResponse({"error": "MIDI not ready"}, status_code=404)

            midi_file = Path(job.midi_path)
            if not midi_file.exists():
                return JSONResponse({"error": "MIDI file not found"}, status_code=404)

            return FileResponse(
                midi_file,
                media_type="audio/midi",
                filename=f"grimlock_{job_id}.mid"
            )

        @app.get("/download/{job_id}/json")
        async def download_json(job_id: str):
            """Download JSON audit file for completed job."""
            job = self._jobs.get(job_id)
            if not job:
                return JSONResponse({"error": "Job not found"}, status_code=404)
            if job.status != JobStatus.COMPLETED or not job.json_path:
                return JSONResponse({"error": "JSON not ready"}, status_code=404)

            json_file = Path(job.json_path)
            if not json_file.exists():
                return JSONResponse({"error": "JSON file not found"}, status_code=404)

            return FileResponse(
                json_file,
                media_type="application/json",
                filename=f"grimlock_{job_id}_audit.json"
            )

        @app.get("/jobs")
        async def list_jobs(limit: int = 50):
            """List recent jobs."""
            jobs = list(self._jobs.values())
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            return JSONResponse([j.to_dict() for j in jobs[:limit]])

        @app.post("/cancel/{job_id}")
        async def cancel_job(job_id: str):
            """Cancel a running job."""
            job = self._jobs.get(job_id)
            if not job:
                return JSONResponse({"error": "Job not found"}, status_code=404)

            if job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
                job.status = JobStatus.CANCELLED
                logger.info(f"Job {job_id} cancelled")
                return JSONResponse({"message": f"Job {job_id} cancelled"})

            return JSONResponse({"message": f"Job {job_id} cannot be cancelled (status: {job.status.value})"})

        # =========================================================
        # WebSocket Endpoint (FIXED: stores connection)
        # =========================================================

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_websocket_client(websocket)

        # =========================================================
        # Start server
        # =========================================================
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def _run_transcription(self, job_id: str):
        """Run transcription in background thread."""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.stage = "initializing"
        job.progress = 5

        start_time = time.time()

        try:
            # Create progress callback with thread-safe messaging
            progress_callback = JobProgressCallback(job, self, self._event_loop)

            # Parse guided params
            guided = None
            if job.guided_params and not job.guided_params.get("error"):
                gp = job.guided_params
                guided = GuidedParams(
                    tempo_bpm=gp.get('tempo_bpm'),
                    time_signature=gp.get('time_signature'),
                    key_signature=gp.get('key_signature'),
                    genre=gp.get('genre_hint')
                )
                if gp.get('time_signature'):
                    guided.time_signature = gp['time_signature']

            # Create MusicBox for this job
            music_box = MusicBox(
                config=MusicBoxConfig(log_path=self._output_dir / f"musicbox_{job_id}.jsonl")
            )
            job.music_box_session = music_box.get_session_id()

            # Create Scribe
            scribe = Scribe(music_box=music_box)

            # Create MemoryGuardian
            guardian = MemoryGuardian(music_box=music_box)

            # Create pipeline with progress callback
            pipeline = GrimlockPipeline(
                guided_params=guided,
                music_box=music_box,
                guardian=guardian,
                scribe=scribe,
                progress_callback=progress_callback  # CRITICAL: Wire the callback
            )

            # Run transcription off the event loop. pipeline.run() is fully
            # synchronous and can take many minutes - calling it directly
            # here would block this coroutine's event loop the entire time
            # (no other WebSocket messages, HTTP requests, or other jobs
            # could be served). JobProgressCallback already assumes this:
            # it uses asyncio.run_coroutine_threadsafe specifically to get
            # progress messages from a background thread back to the loop.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._executor, pipeline.run, job.audio_path)

            # Extract results
            notes = result.notes if hasattr(result, 'notes') else []
            verdict = result.verdict if hasattr(result, 'verdict') else {}
            vetoes = verdict.get('hard_vetoes', [])
            vetoes.extend(verdict.get('soft_vetoes', []))

            # Update job with results
            job.notes = [progress_callback._note_to_dict(n) for n in notes[:500]]
            job.veto_count = len(vetoes)
            job.vetoes = vetoes
            job.avg_confidence = sum(n.confidence for n in notes) / len(notes) if notes else 0
            job.elapsed_sec = time.time() - start_time
            job.result = result

            # Export MIDI
            from export.midi_writer import MidiWriter
            midi_path = self._output_dir / f"{job_id}_transcription.mid"
            MidiWriter().export_midi(events=notes, output_path=str(midi_path))
            job.midi_path = str(midi_path)

            # Export JSON audit
            audit_data = {
                "job_id": job_id,
                "original_filename": job.original_filename,
                "guided_params": job.guided_params,
                "notes": job.notes,
                "vetoes": vetoes,
                "forensic_markers": job.forensic_markers,
                "verdict": verdict,
                "elapsed_sec": job.elapsed_sec,
                "music_box_session": job.music_box_session
            }
            json_path = self._output_dir / f"{job_id}_audit.json"
            with open(json_path, 'w') as f:
                json.dump(audit_data, f, indent=2, default=str)
            job.json_path = str(json_path)

            # Finalize
            music_box.end_session()

            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.stage = "done"
            job.completed_at = datetime.now()

            logger.info(f"Job {job_id} completed: {len(notes)} notes, {job.elapsed_sec:.1f}s")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.error(f"Job {job_id} failed: {e}")

        finally:
            try:
                temp_path = Path(job.audio_path)
                if temp_path.exists() and temp_path.parent == self._temp_dir:
                    temp_path.unlink()
            except Exception:
                pass

    async def _handle_websocket_client(self, websocket: WebSocket):
        """Handle WebSocket client connection for real-time streaming."""
        session_id = str(uuid.uuid4())[:8]

        # FIXED: Store the actual WebSocket connection
        session = ClientSession(
            session_id=session_id,
            websocket=websocket,
            connected_at=datetime.now(),
            last_activity=datetime.now()
        )
        self._sessions[session_id] = session

        logger.info(f"WebSocket client {session_id} connected")

        try:
            # Send ready message with session ID
            await session.send_json({
                "type": MessageType.READY.value,
                "session_id": session_id,
                "config": {
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "supported_formats": ["pcm_f32le"],
                    "max_chunk_size": 8192
                }
            })

            # Send status of any existing jobs for this client (reconnection support)
            if session.current_job_id and session.current_job_id in self._jobs:
                job = self._jobs[session.current_job_id]
                await session.send_json({
                    "type": MessageType.JOB_UPDATE.value,
                    "job_id": session.current_job_id,
                    "stage": job.stage,
                    "progress": job.progress,
                    "status": job.status.value
                })

            while True:
                try:
                    message = await websocket.receive_json()
                    await self._process_websocket_message(session, message)
                except WebSocketDisconnect:
                    break

        except Exception as e:
            logger.error(f"WebSocket error with {session_id}: {e}")
        finally:
            if session_id in self._sessions:
                del self._sessions[session_id]
            logger.info(f"WebSocket client {session_id} disconnected")

    async def _process_websocket_message(
            self,
            session: ClientSession,
            message: Dict
    ):
        """Process WebSocket message."""
        msg_type = message.get("type")

        if msg_type == MessageType.PING.value:
            await session.send_json({"type": MessageType.PONG.value})

        elif msg_type == MessageType.CONFIGURE.value:
            session.config.update(message.get("config", {}))
            await session.send_json({
                "type": MessageType.STATUS.value,
                "status": "configured",
                "config": session.config
            })

        elif msg_type == MessageType.AUDIO_START.value:
            session.clear_audio()
            await session.send_json({
                "type": MessageType.STATUS.value,
                "status": "streaming_started"
            })

        elif msg_type == MessageType.AUDIO_CHUNK.value:
            audio_b64 = message.get("audio", "")
            if audio_b64:
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    session.add_audio_chunk(audio_np)

                    if session.config.get("real_time", False):
                        await session.send_json({
                            "type": MessageType.TRANSCRIPTION_CHUNK.value,
                            "chunk_id": session.chunk_count,
                            "duration_sec": session.get_duration_seconds(),
                            "chunk_size": len(audio_np)
                        })

                    session.chunk_count += 1
                except Exception as e:
                    await session.send_json({
                        "type": MessageType.ERROR.value,
                        "error": f"Invalid audio chunk: {e}"
                    })

        elif msg_type == MessageType.AUDIO_END.value:
            full_audio = session.get_full_audio()
            duration = session.get_duration_seconds()

            if len(full_audio) > 0:
                job_id = str(uuid.uuid4())[:8]

                temp_path = self._temp_dir / f"{job_id}_stream.wav"
                if SF_AVAILABLE:
                    sf.write(temp_path, full_audio, TARGET_SAMPLE_RATE)
                else:
                    np.save(temp_path.with_suffix('.npy'), full_audio)

                job = TranscriptionJob(
                    job_id=job_id,
                    audio_path=str(temp_path),
                    original_filename=f"stream_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                    guided_params=session.config.get("guided_params"),
                    status=JobStatus.PENDING
                )
                self._jobs[job_id] = job
                self._job_sessions[job_id] = session.session_id
                session.current_job_id = job_id

                asyncio.create_task(self._run_transcription(job_id))

                await session.send_json({
                    "type": MessageType.TRANSCRIPTION_FINAL.value,
                    "job_id": job_id,
                    "duration_sec": duration,
                    "message": "Streaming complete, transcription started"
                })
            else:
                await session.send_json({
                    "type": MessageType.ERROR.value,
                    "error": "No audio data received"
                })

            session.clear_audio()

        elif msg_type == MessageType.RESUME_JOB.value:
            # Support for reconnecting to an existing job
            job_id = message.get("job_id")
            if job_id and job_id in self._jobs:
                self._job_sessions[job_id] = session.session_id
                session.current_job_id = job_id

                job = self._jobs[job_id]
                await session.send_json({
                    "type": MessageType.JOB_UPDATE.value,
                    "job_id": job_id,
                    "stage": job.stage,
                    "progress": job.progress,
                    "notes": job.notes[:20],
                    "veto_count": job.veto_count,
                    "status": job.status.value
                })

                logger.info(f"Session {session.session_id} resumed job {job_id}")

    # FIXED: _send_to_session now actually works
    async def _send_to_session(self, session_id: str, message: Dict):
        """Send message to specific session with connection validation."""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            await session.send_json(message)
        else:
            logger.debug(f"Session {session_id} not found, message dropped")

    def _get_classic_interface_html(self) -> str:
        """Return the classic Grimlock interface HTML."""
        return self._get_embedded_html()

    def _get_embedded_html(self) -> str:
        """Get embedded HTML for standalone server with WebSocket support."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grimlock 5.0 · Classic Transcription Workstation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0c0b14; color: #e8e2d6; font-family: 'Segoe UI', monospace; padding: 24px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: #12101c; border-left: 6px solid #c47f2e; padding: 20px 28px; margin-bottom: 32px; border-radius: 12px; }
        .header h1 { font-size: 2rem; background: linear-gradient(130deg, #e4b87a, #c47f2e); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .laws { display: flex; gap: 12px; margin-top: 12px; }
        .law { background: #1e1b2a; padding: 4px 14px; border-radius: 30px; font-size: 0.7rem; border-left: 3px solid #c47f2e; }
        .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        @media (max-width: 880px) { .main-grid { grid-template-columns: 1fr; } }
        .panel { background: #12101c; border-radius: 16px; border: 1px solid #2e2a3a; padding: 22px; }
        .panel h2 { color: #e4b87a; margin-bottom: 18px; border-bottom: 1px solid #2a2538; padding-bottom: 8px; }
        .upload-area { background: #0a0912; border: 2px dashed #3e3650; border-radius: 20px; padding: 32px; text-align: center; cursor: pointer; margin-bottom: 24px; }
        .upload-area:hover { border-color: #c47f2e; background: #161222; }
        .guided-group { background: #0a0912; border-radius: 14px; padding: 16px; margin: 20px 0; border-left: 4px solid #c47f2e; }
        .guided-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px; }
        .field-group { display: flex; flex-direction: column; gap: 5px; }
        .field-group label { font-size: 0.7rem; color: #9c8e78; }
        input, select { background: #1a1727; border: 1px solid #3a324a; color: #f0e6d8; padding: 8px 12px; border-radius: 10px; }
        button { background: #262137; border: none; color: #e8e2d6; padding: 10px 24px; border-radius: 40px; cursor: pointer; margin-right: 10px; }
        button.primary { background: #c47f2e; color: #0c0b14; font-weight: bold; }
        .progress-card { background: #0a0912; border-radius: 14px; padding: 14px; margin-top: 20px; }
        .progress-bar { height: 6px; background: #2a2438; border-radius: 4px; margin: 10px 0; overflow: hidden; }
        .progress-fill { width: 0%; height: 100%; background: #c47f2e; transition: width 0.2s; }
        .notes-pane { background: #07060c; border-radius: 14px; padding: 12px; height: 350px; overflow-y: auto; }
        .note-item { background: #12101c; margin-bottom: 8px; padding: 8px 12px; border-radius: 12px; border-left: 3px solid #c47f2e; display: flex; gap: 12px; flex-wrap: wrap; }
        .log-box { background: #07060c; border-radius: 14px; padding: 14px; height: 250px; overflow-y: auto; font-size: 0.75rem; }
        .log-entry { padding: 4px 8px; border-left: 3px solid; margin-bottom: 4px; }
        .log-info { border-left-color: #6a7c8f; }
        .log-success { border-left-color: #c47f2e; color: #e4b87a; }
        .forensic-marker { border-left-color: #c44f2e; background: rgba(196, 79, 46, 0.1); }
        .footer { margin-top: 32px; text-align: center; font-size: 0.7rem; color: #6c617a; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>◷ GRIMLOCK 5.0</h1>
        <div class="subtitle">classic transcription · guided mode · epistemic veto</div>
        <div class="laws"><span class="law">⚡ RESOURCE SURVIVAL</span><span class="law">📜 SCRIBE'S TRUTH</span><span class="law">🔄 RELATIONAL PHYSICS</span></div>
    </div>
    <div class="main-grid">
        <div class="panel">
            <h2>📁 UPLOAD</h2>
            <div id="uploadArea" class="upload-area" style="cursor:pointer;">🎚️ click or drag audio<br><small>WAV, MP3, FLAC, M4A</small></div>
            <input type="file" id="fileInput" accept="audio/*" style="display:none;">
            <div class="guided-group"><label><input type="checkbox" id="guidedEnable"> 🎯 GUIDED MODE</label>
            <div id="guidedFields" style="display:none;">
                <div class="guided-fields">
                    <div class="field-group"><label>TEMPO (BPM)</label><input type="number" id="tempoInput" placeholder="auto"></div>
                    <div class="field-group"><label>TIME SIG</label><div style="display:flex;gap:6px;"><input type="number" id="timeNum" placeholder="4" style="width:65px;"><span>/</span><input type="number" id="timeDen" placeholder="4" style="width:65px;"></div></div>
                    <div class="field-group"><label>KEY</label><select id="keySelect"><option value="">auto</option><option>C major</option><option>C minor</option><option>D major</option><option>E minor</option><option>F major</option><option>G major</option><option>A minor</option></select></div>
                    <div class="field-group"><label>GENRE</label><select id="genreHint"><option value="">auto</option><option>jazz</option><option>rock</option><option>classical</option><option>electronic</option></select></div>
                </div>
            </div>
            </div>
            <div><button id="transcribeBtn" class="primary">🎙️ TRANSCRIBE</button><button id="clearBtn">🗑️ CLEAR</button></div>
            <div class="progress-card"><div class="progress-bar"><div id="progressFill" class="progress-fill"></div></div><div><span id="stageName">idle</span> · <span id="noteCount">0 notes</span> · vetoes: <span id="vetoCount">0</span></div></div>
        </div>
        <div class="panel">
            <h2>📜 RESULTS</h2>
            <div id="notesDisplay" class="notes-pane"><div style="text-align:center;color:#7e6e56;">upload a track → click TRANSCRIBE</div></div>
            <div style="margin-top:16px;"><button id="downloadMidiBtn" disabled>🎵 MIDI</button><button id="downloadJsonBtn" disabled>📄 JSON audit</button></div>
        </div>
    </div>
    <div class="panel"><h2>📡 MUSIC BOX / FORENSIC LOG</h2>
        <div id="logArea" class="log-box"></div>
        <div id="forensicArea" style="margin-top: 10px; font-size: 0.7rem; color: #c44f2e;"></div>
    </div>
    <div class="footer">Law of Scribe's Truth — non‑destructive audit — Mel-RoFormer + Basic Pitch — 5 Strategies active</div>
</div>
<script>
    let activeJobId = null, pollInterval = null;
    let ws = null;
    let sessionId = null;

    const uploadArea = document.getElementById('uploadArea'), fileInput = document.getElementById('fileInput');
    const transcribeBtn = document.getElementById('transcribeBtn'), clearBtn = document.getElementById('clearBtn');
    const downloadMidiBtn = document.getElementById('downloadMidiBtn'), downloadJsonBtn = document.getElementById('downloadJsonBtn');
    const notesDisplay = document.getElementById('notesDisplay'), logArea = document.getElementById('logArea');
    const progressFill = document.getElementById('progressFill'), stageNameSpan = document.getElementById('stageName');
    const noteCountSpan = document.getElementById('noteCount'), vetoCountSpan = document.getElementById('vetoCount');
    const guidedEnable = document.getElementById('guidedEnable'), guidedFields = document.getElementById('guidedFields');
    const forensicArea = document.getElementById('forensicArea');
    let selectedFile = null;

    // WebSocket connection for real-time updates
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

        ws.onopen = () => {
            addLog('🔌 WebSocket connected — real-time progress enabled', 'success');

            // Resume existing job if we have one
            if (activeJobId) {
                ws.send(JSON.stringify({
                    type: 'resume_job',
                    job_id: activeJobId
                }));
            }
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            switch(msg.type) {
                case 'ready':
                    sessionId = msg.session_id;
                    addLog(`✓ Session ${sessionId} ready`, 'success');
                    break;

                case 'progress':
                    if (msg.job_id === activeJobId) {
                        stageNameSpan.innerText = msg.stage;
                        progressFill.style.width = msg.progress + '%';
                        if (msg.note_count > 0) {
                            noteCountSpan.innerText = msg.note_count + ' notes';
                        }
                        if (msg.confidence) {
                            // Update confidence display
                        }
                        if (msg.veto_count !== undefined) {
                            vetoCountSpan.innerText = msg.veto_count;
                        }
                    }
                    break;

                case 'job_update':
                    if (msg.job_id === activeJobId) {
                        stageNameSpan.innerText = msg.stage || msg.status;
                        progressFill.style.width = (msg.progress || 0) + '%';
                        if (msg.notes) renderNotes(msg.notes, msg.veto_count || 0);
                        if (msg.status === 'completed') {
                            addLog(`✅ Job ${msg.job_id} completed!`, 'success');
                            downloadMidiBtn.disabled = false;
                            downloadJsonBtn.disabled = false;
                        } else if (msg.status === 'failed') {
                            addLog(`❌ Job ${msg.job_id} failed`, 'error');
                        }
                    }
                    break;

                case 'stage_complete':
                    addLog(`✓ Stage "${msg.completed_stage}" complete`, 'info');
                    break;

                case 'forensic_marker':
                    if (msg.job_id === activeJobId) {
                        const marker = msg.marker;
                        addLog(`⚠️ Forensic marker at ${marker.time_sec.toFixed(1)}s: ${marker.reason} (conf: ${marker.confidence.toFixed(2)})`, 'warn');
                        const markerDiv = document.createElement('div');
                        markerDiv.className = 'log-entry forensic-marker';
                        markerDiv.innerHTML = `[${new Date().toLocaleTimeString()}] 🔍 ${marker.time_sec.toFixed(1)}s: ${marker.reason}`;
                        forensicArea.appendChild(markerDiv);
                    }
                    break;

                case 'error':
                    addLog(`❌ Error: ${msg.error}`, 'error');
                    break;

                case 'pong':
                    // Keepalive response
                    break;
            }
        };

        ws.onclose = () => {
            addLog('🔌 WebSocket disconnected — reconnecting in 3s...', 'warn');
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (err) => {
            addLog(`WebSocket error: ${err}`, 'error');
        };
    }

    guidedEnable.onchange = () => guidedFields.style.display = guidedEnable.checked ? 'block' : 'none';
    uploadArea.onclick = () => fileInput.click();
    fileInput.onchange = (e) => { if(e.target.files.length) { selectedFile = e.target.files[0]; addLog('✓ selected: ' + selectedFile.name, 'success'); } };

    function addLog(msg, type='info') {
        const d = document.createElement('div');
        d.className = `log-entry log-${type}`;
        d.innerHTML = `[${new Date().toLocaleTimeString()}] ${msg}`;
        logArea.appendChild(d);
        d.scrollIntoView({block:'nearest'});
    }

    function renderNotes(notes, veto) {
        if(!notes || !notes.length) {
            notesDisplay.innerHTML = '<div style="text-align:center;color:#7e6e56;">no notes transcribed</div>';
            noteCountSpan.innerText = '0 notes';
            return;
        }
        noteCountSpan.innerText = notes.length + ' notes';
        vetoCountSpan.innerText = veto;
        let html = '';
        for(let n of notes.slice(0,80)) {
            const confidenceClass = n.confidence < 0.5 ? 'badge-low' : (n.confidence < 0.8 ? 'badge-med' : 'badge-high');
            html += `<div class="note-item"><span>🎵 ${n.pitch_name||'?'}</span><span>⏱️ ${(n.start_ms/1000).toFixed(2)}s</span><span>💪 vel ${n.velocity}</span><span class="${confidenceClass}">${Math.round(n.confidence*100)}%</span></div>`;
        }
        if(notes.length>80) html += `<div>... ${notes.length-80} more</div>`;
        notesDisplay.innerHTML = html;
    }

    async function startTranscription() {
        if(!selectedFile) { addLog('⚠️ select audio file first','error'); return; }
        if(pollInterval) clearInterval(pollInterval);
        transcribeBtn.disabled = true;

        // Connect WebSocket first
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            connectWebSocket();
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        let guided = null;
        if(guidedEnable.checked) {
            guided = {
                tempo_bpm: parseInt(document.getElementById('tempoInput').value) || null,
                time_signature: {
                    numerator: parseInt(document.getElementById('timeNum').value) || null,
                    denominator: parseInt(document.getElementById('timeDen').value) || null
                },
                key_signature: document.getElementById('keySelect').value || null,
                genre_hint: document.getElementById('genreHint').value || null
            };
        }
        const fd = new FormData();
        fd.append('audio', selectedFile);
        if(guided) fd.append('guided', JSON.stringify(guided));
        addLog('⏳ uploading...','info');
        progressFill.style.width = '5%';
        stageNameSpan.innerText = 'uploading';
        try {
            const res = await fetch('/upload', { method: 'POST', body: fd });
            const data = await res.json();
            if(data.job_id) {
                activeJobId = data.job_id;
                addLog(`job ${activeJobId} started — streaming windows + Mel-RoFormer`,'success');
                // WebSocket will handle progress updates
            } else throw new Error('no job id');
        } catch(e) { addLog('upload error: '+e.message,'error'); transcribeBtn.disabled=false; }
    }

    downloadMidiBtn.onclick = () => { if(activeJobId) window.location.href = `/download/${activeJobId}/midi`; };
    downloadJsonBtn.onclick = () => { if(activeJobId) window.location.href = `/download/${activeJobId}/json`; };
    clearBtn.onclick = () => {
        notesDisplay.innerHTML = '<div style="text-align:center;color:#7e6e56;">cleared</div>';
        forensicArea.innerHTML = '';
        if(activeJobId) fetch(`/cancel/${activeJobId}`, { method: 'POST' });
        activeJobId = null;
        transcribeBtn.disabled = false;
        downloadMidiBtn.disabled = true;
        downloadJsonBtn.disabled = true;
        progressFill.style.width = '0%';
        stageNameSpan.innerText = 'idle';
        addLog('🧹 cleared','info');
    };
    transcribeBtn.onclick = startTranscription;

    // Connect WebSocket on page load
    connectWebSocket();

    addLog('Grimlock 5.0 ready — guided mode | Mel-RoFormer | 5 Strategies | Live WebSocket','success');
</script>
</body>
</html>"""

    async def _start_websockets(self):
        """Fallback: pure WebSocket server (no HTTP upload)."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets not available. Install with: pip install websockets")

        async with serve(self._handle_websocket_client_raw, self.host, self.port):
            await asyncio.Future()

    async def _handle_websocket_client_raw(self, websocket):
        """Handle raw websocket client."""
        await self._handle_websocket_client(websocket)

    def stop(self):
        """Stop the server."""
        self._running = False
        self._executor.shutdown(wait=True)

        # Close all WebSocket connections
        for session_id, session in self._sessions.items():
            if session.websocket and not getattr(session.websocket, 'closed', False):
                asyncio.create_task(session.websocket.close())

        logger.info("Server stopped")


# ========================================================================
# Main Entry Point
# ========================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Grimlock 5.0 Transcription Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--no-fastapi", action="store_true", help="Use pure websockets (no HTTP upload)")
    parser.add_argument("--max-jobs", type=int, default=4, help="Maximum concurrent jobs")
    args = parser.parse_args()

    server = WebSocketServer(
        host=args.host,
        port=args.port,
        use_fastapi=not args.no_fastapi,
        max_concurrent_jobs=args.max_jobs
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop()


if __name__ == "__main__":
    asyncio.run(main())