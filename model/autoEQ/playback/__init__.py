"""Playback pipeline — consumes a timeline.json from infer_pseudo and applies
per-scene EQ with boundary crossfades, then remuxes with the original video.

Stages (spec V3.3 §5-9 = V3.2 §5-12..5-14):
  1. extract original audio (ffmpeg) preserving channels + sample rate
  2. for each scene in timeline.scenes → build pedalboard 10-band biquad chain
     from `eq_preset.effective_bands`
  3. apply EQ to that scene's audio region
  4. raised-cosine crossfade at scene boundaries (200-500ms)
  5. write processed audio wav
  6. ffmpeg remux: video stream copied lossless, audio re-encoded AAC 192kbps
"""
