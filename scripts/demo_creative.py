# Copyright 2026 Tobias Østen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quick demo of the creative stack."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BrainController

bc = BrainController()
bc.boot_system()

inputs = [
    "I lost my dog yesterday. He was my best friend for 14 years.",
    "I just got promoted at work! Best day ever!",
    "I am furious! My coworker stole my work!",
    "What is the meaning of life? Do we have free will?",
]

for text in inputs:
    r = bc.generate_creative_output(text)
    p = r["palette"]
    m = r["music_prompt"]
    q = r["quality"]
    print("=" * 70)
    print("INPUT:   ", text[:60])
    print("EMOTION: ", r["state"].emotion)
    print("QUALITY: ", "score=%.2f passed=%s rec=%s" % (q.score, q.passed, q.recommendation))
    print("PALETTE: ", "%s | %dBPM | %s | energy=%.2f | valence=%.2f" % (p.genre, p.tempo_bpm, p.key, p.energy, p.valence))
    print("MOOD:    ", ", ".join(p.mood_tags[:5]))
    print("INSTR:   ", ", ".join(p.instruments[:5]))
    print("REFS:    ", ", ".join(p.reference_artists[:3]))
    print("STYLE:   ", m.style_tags[:80] + "...")
    print("NEG:     ", m.negative_prompt)
    if m.lyrics:
        lines = m.lyrics.split("\n")
        print("LYRICS:  ", lines[0])
        if len(lines) > 1:
            print("         ", lines[1])
        if len(lines) > 4:
            print("         ... " + lines[4])
    print()
