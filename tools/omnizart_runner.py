"""Standalone Omnizart runner - executed by the ISOLATED Python 3.9 / TF 2.5
conda env (`omnizart-tf25`), never by Symphony's own interpreter.

Omnizart 0.6.0's music model is a pre-TF-2.5 SavedModel that computes a
degenerate all-background posteriorgram under modern TensorFlow (Symphony
runs TF 2.15 for Basic Pitch / SPICE / CREPE). Rather than downgrade the
whole stack, Symphony shells out to this script in an env with the TF the
model was built for; the two never share a process. Contract:

    <omni_env_python> omnizart_runner.py <input_wav> <output_json> [model]

Writes {"notes": [{pitch,start,end,velocity,instrument}, ...]} to
<output_json>. Exits non-zero with a JSON {"error": ...} on failure so the
caller can log and fall back to an empty witness.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # force CPU; deterministic
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
import json


def main():
    if len(sys.argv) < 3:
        json.dump({"error": "usage: omnizart_runner.py <wav> <json_out> [model]"},
                  sys.stdout)
        return 2
    wav_in = sys.argv[1]
    json_out = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "Pop"
    try:
        from omnizart.music import app as omni
        out_dir = os.path.dirname(os.path.abspath(json_out)) or "."
        midi = omni.transcribe(wav_in, model_path=model, output=out_dir)
        notes = []
        for inst in midi.instruments:
            for n in inst.notes:
                notes.append({
                    "pitch": int(n.pitch),
                    "start": float(n.start),
                    "end": float(n.end),
                    "velocity": int(n.velocity),
                    "instrument": getattr(inst, "name", "") or "",
                })
        with open(json_out, "w") as f:
            json.dump({"notes": notes, "model": model}, f)
        sys.stderr.write(f"[omnizart_runner] {len(notes)} notes -> {json_out}\n")
        return 0
    except Exception as e:  # noqa: BLE001 - report everything to the caller
        try:
            with open(json_out, "w") as f:
                json.dump({"error": f"{type(e).__name__}: {e}"}, f)
        except Exception:
            pass
        sys.stderr.write(f"[omnizart_runner] FAILED: {type(e).__name__}: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
