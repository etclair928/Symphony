================================================================================
                           GRIMLOCK 5.0
              Audio Transcription with Epistemic Veto
================================================================================

Version: 5.0.0
Status: Production Ready
Python: 3.8+

================================================================================
TABLE OF CONTENTS
================================================================================

1.  Overview
2.  The Three Laws of Grimlock
3.  Installation
4.  Quick Start
5.  Command Line Usage
6.  Python API
7.  Architecture
8.  WebSocket Server
9.  Configuration
10. Troubleshooting
11. License

================================================================================
1. OVERVIEW
================================================================================

Grimlock 5.0 is an intelligent audio transcription system that converts audio
recordings (WAV, MP3, FLAC, etc.) into MIDI and JSON representations. Unlike
traditional transcription systems, Grimlock applies:

- EPISTEMIC VETO: Any qualified witness can declare a falsehood
- RELATIONAL PHYSICS: Measures distance between notes, not to a grid
- NON-DESTRUCTIVE AUDIT: All transformations are reversible
- MEMORY SURVIVAL: Staggered deletion prevents memory exhaustion

The system is designed for transcription of:

- Bass guitar lines (jazz, funk, rock)
- Drum tracks (kick, snare, hi-hat, cymbals)
- Melodic instruments (piano, guitar, monophonic lines)
- Full mixes with source separation

================================================================================
2. THE THREE LAWS OF GRIMLOCK
================================================================================

LAW 1: RESOURCE SURVIVAL (The 4.5 Legacy)
-----------------------------------------
"The math must never exceed the machine."

- Mono-downmixing on initial load (halves RAM footprint)
- Staggered deletion between agents
- Downsampled analysis (16kHz) - jazz bass doesn't need 48kHz

LAW 2: UNIDIRECTIONAL INTEGRITY (The 4.7 Legacy)
------------------------------------------------
"Data flows forward; dependencies never look back."

- Types are foundation and cannot import from anything else
- Agents perform single tasks (Separate, Detect, or Analyze)
- The Orchestrator (Pipeline) is the only module that touches everything

LAW 3: THE SCRIBE'S TRUTH
-------------------------
"Confidence must be earned, not assumed."

- If the Pitch Engine returns 90% silence, flag low confidence
- Schoenberg Mirror validates harmonic series before accepting notes
- Every transcription passes validation gate before finalization

================================================================================
3. INSTALLATION
================================================================================

3.1 Basic Installation
----------------------
$ pip install -e .

3.2 Full Installation (All Features)
------------------------------------
$ pip install -e .[full]

3.3 Development Installation
----------------------------
$ pip install -e .[dev]

3.4 Manual Dependencies
-----------------------
Core (always required):
  - numpy>=1.19.0
  - scipy>=1.5.0
  - librosa>=0.8.0
  - soundfile>=0.10.0
  - psutil>=5.8.0

Optional (better quality):
  - torch>=1.9.0 (for neural models)
  - demucs>=3.0.0 (source separation)
  - basic-pitch>=0.3.0 (pitch detection)
  - madmom>=0.16.1 (beat tracking)
  - fastapi>=0.68.0 (WebSocket server)
  - uvicorn>=0.15.0 (WebSocket server)

================================================================================
4. QUICK START
================================================================================

4.1 Command Line
----------------
# Basic transcription to MIDI
$ python main.py my_audio.wav

# With JSON export
$ python main.py my_audio.wav --json --pretty

# Specify output directory
$ python main.py my_audio.wav -o ./output

# Fast mode (lower quality, less memory)
$ python main.py my_audio.wav --fast

4.2 Python API
--------------
from grimlock_5 import transcribe

# Simple one-line transcription
result = transcribe("bass_line.wav", "output.mid")

# Advanced usage
from grimlock_5 import GrimlockPipeline
from grimlock_5.core.types import ExportOptions

pipeline = GrimlockPipeline()
result = pipeline.transcribe("drum_loop.wav", "output.mid")

print(f"Transcribed {result['verdict']['stages_completed']} stages")
print(f"Trustworthy: {result['verdict']['is_trustworthy']}")

================================================================================
5. COMMAND LINE USAGE
================================================================================

5.1 Basic Syntax
----------------
python main.py INPUT [INPUT...] [OPTIONS]

5.2 Input Options
-----------------
INPUT                     Audio file(s) (WAV, MP3, FLAC, M4A, OGG, AIFF)
--batch                   Process all files in directories recursively
--extensions EXT          Extensions for batch mode (default: .wav,.mp3,.flac,.m4a)

5.3 Output Options
------------------
-o, --output-dir DIR      Output directory (default: output)
--output-name NAME        Custom output filename (without extension)
--midi                    Export MIDI file (default: True)
--no-midi                 Disable MIDI export
--json                    Export JSON with forensic audit
--pretty                  Pretty-print JSON output
--separate-tracks         Export separate MIDI tracks per source

5.4 Processing Options
----------------------
--sample-rate HZ          Target sample rate (default: 16000)
--no-mono                 Disable mono downmixing (increases memory)
--no-octave-restore       Disable octave restoration for bass
--no-heartbeat            Disable heartbeat note for empty transcriptions
--fast                    Fast mode (lower quality, less memory)
--skip-separation         Skip source separation (use raw audio)
--skip-quantization       Skip quantization (preserve original timing)

5.5 Forensic Options
--------------------
--music-box PATH          Path for MusicBox forensic log file
-v, --verbose             Verbose output (show stage details)
-q, --quiet               Quiet mode (minimal output)

5.6 Advanced Options
--------------------
--config FILE             JSON configuration file
--device auto|cpu|cuda    Compute device (default: auto)
--max-memory MB           Maximum memory in MB (default: 2048)

5.7 Examples
------------
# Single file with all exports
python main.py guitar.wav --json --pretty --separate-tracks

# Batch process all WAV files
python main.py ./songs/ --batch --extensions .wav

# Fast mode for long file
python main.py long_recording.wav --fast --no-json

# With custom output
python main.py input.wav -o ./my_transcriptions --output-name transcription1

================================================================================
6. PYTHON API
================================================================================

6.1 Quick Transcription
------------------------
def transcribe(audio_path, output_path=None, export_midi=True,
               export_json=False, **kwargs)

# Example
result = transcribe("audio.wav", export_json=True, octave_restore=True)

6.2 GrimlockPipeline
--------------------
from grimlock_5 import GrimlockPipeline

pipeline = GrimlockPipeline()
result = pipeline.transcribe("audio.wav", "output.mid")

# Access results
notes = pipeline.get_all_notes()
groove = pipeline.get_groove_field()
tempo = pipeline.get_tempo_map()

6.3 Component Access
--------------------
from grimlock_5 import (
    MusicBox,           # Forensic flight recorder
    Scribe,             # Epistemic veto authority
    MemoryGuardian,     # Memory management
    AgentFactory        # Agent creation
)

# Create custom pipeline
music_box = MusicBox()
scribe = Scribe(music_box)
guardian = MemoryGuardian(music_box)
factory = AgentFactory({"device": "cuda"})

pipeline = GrimlockPipeline(
    music_box=music_box,
    guardian=guardian,
    scribe=scribe,
    agent_factory=factory
)

6.4 Working with Results
------------------------
result = pipeline.transcribe("audio.wav")

# Check trustworthiness
if result['verdict']['is_trustworthy']:
    print("Transcription is trustworthy")
else:
    print(f"Vetoes: {result['verdict']['veto_count']}")

# Access timing
timing = result['timing']
for stage, ms in timing.items():
    print(f"{stage}: {ms:.1f}ms")

# Get veto details
for veto in result['verdict']['hard_vetoes']:
    print(f"Veto by {veto['source']}: {veto['reason']}")

================================================================================
7. ARCHITECTURE
================================================================================

7.1 Directory Structure
------------------------
grimlock_5/
├── core/           # Types, constants, protocols (bedrock)
├── memory/         # Guardian and Arena (resource management)
├── orchestration/  # Pipeline, Scribe, MusicBox
├── agents/         # All processing agents
│   ├── separation/    # Demucs, BS-Roformer, Hybrid
│   ├── detection/     # Rhythm, Pitch, Tonal, Drum
│   ├── analysis/      # Groove, Voice, Tempo
│   ├── quantization/  # Ritornello, Velocity Merge
│   └── validation/    # Consensus Engine
├── ingestion/      # Audio loading and downsampling
├── export/         # MIDI and JSON writers
├── server/         # WebSocket server
└── main.py         # CLI entry point

7.2 Pipeline Stages (Linear)
----------------------------
1. INGESTION    - Load audio (mono, 16kHz)
2. SEPARATION   - Demucs/BS-Roformer to split stems
3. DETECTION    - Rhythm, Pitch, Tonal, Drum
4. ANALYSIS     - Groove Field, Voice, Tempo
5. QUANTIZATION - Ritornello (non-destructive)
6. VALIDATION   - Scribe final gate
7. EXPORT       - MIDI and JSON output

7.3 Memory Flow
---------------
Source Buffer → Mono Downmix → 16kHz Resample → Staggered Delete
                                    ↓
                            Separation (Drums, Bass, Other)
                                    ↓
                    Staggered GC → Next Agent → Staggered GC

================================================================================
8. WEBSOCKET SERVER
================================================================================

8.1 Starting the Server
------------------------
# Using the CLI
$ grimlock-server --host localhost --port 8765

# Or directly
$ python -m server.websocket --host 0.0.0.0 --port 8765

8.2 Client Connection
---------------------
WebSocket endpoint: ws://localhost:8765/ws

8.3 Message Protocol
--------------------
Client -> Server:
- {"type": "audio_start"}                    # Start streaming
- {"type": "audio_chunk", "audio": "base64"} # Audio data
- {"type": "audio_end"}                      # End streaming
- {"type": "ping"}                           # Keep alive
- {"type": "configure", "config": {...}}     # Update config

Server -> Client:
- {"type": "ready", "session_id": "..."}     # Connection ready
- {"type": "transcription_chunk", ...}       # Incremental results
- {"type": "transcription_final", ...}       # Final transcription
- {"type": "status", ...}                    # Status update
- {"type": "error", ...}                     # Error message

8.4 Test Client
---------------
Open browser to http://localhost:8765/ for built-in test client

================================================================================
9. CONFIGURATION
================================================================================

9.1 JSON Config File
--------------------
{
    "sample_rate": 16000,
    "force_mono": true,
    "octave_restore": true,
    "export_midi": true,
    "export_json": true,
    "separate_tracks": false,
    "detect_tempo_changes": true,
    "preserve_swing": true,
    "device": "auto",
    "max_memory_mb": 2048
}

9.2 Environment Variables
--------------------------
GRIMLOCK_SAMPLE_RATE     # Target sample rate (default: 16000)
GRIMLOCK_MEMORY_LIMIT    # Memory limit in MB (default: 2048)
GRIMLOCK_DEVICE          # Device: auto/cpu/cuda (default: auto)
GRIMLOCK_LOG_LEVEL       # DEBUG, INFO, WARNING, ERROR

================================================================================
10. TROUBLESHOOTING
================================================================================

10.1 Common Issues
------------------
Q: Out of memory error
A: Use --fast mode or increase --max-memory

Q: No MIDI file generated
A: Check if audio has detectable pitch content (use --json to see confidence)

Q: Transcription has many errors
A: Try --separate-tracks for cleaner stem separation

Q: WebSocket connection fails
A: Install websockets: pip install websockets fastapi uvicorn

10.2 Logs
---------
# Forensic logs (all decisions)
music_box.jsonl

# Console logs
--verbose for detailed stage timing

10.3 Performance Tuning
-----------------------
# Low memory (<4GB RAM)
--fast --no-mono --skip-separation

# High quality (>8GB RAM)
--device cuda --no-fast

# Batch processing
--batch --extensions .wav

================================================================================
11. LICENSE
================================================================================

MIT License

Copyright (c) 2026 Grimlock Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

================================================================================
                              END OF README
================================================================================