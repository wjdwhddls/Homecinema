"""
roomplan JSON → 탑뷰 PNG (base64)
좌표계: xRIR (x: 가로, y: 앞뒤, z: 높이)
"""
from __future__ import annotations
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_topview(roomplan_json, listener_pos, speaker_positions):
    walls   = roomplan_json.get("walls", [])
    objects = roomplan_json.get("objects", [])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_aspect('equal')

    # ── 벽 (RoomPlan x-z → xRIR x-y 변환: y = -z)
    for wall in walls:
        m = np.array(wall["transform"], dtype=float).reshape(4, 4, order="F")
        dims = wall["dimensions"]
        half_w = float(dims[0]) * 0.5
        local_pts = np.array([
            [-half_w, 0.0, 0.0, 1.0],
            [+half_w, 0.0, 0.0, 1.0],
        ])
        world_pts = (m @ local_pts.T).T[:, :3]
        xs = [world_pts[0][0],  world_pts[1][0]]
        ys = [-world_pts[0][2], -world_pts[1][2]]
        ax.plot(xs, ys, color='#e0e0e0', linewidth=3, solid_capstyle='round')

    # ── 가구 (마찬가지로 변환)
    for obj in objects:
        m = np.array(obj["transform"], dtype=float).reshape(4, 4, order="F")
        dims = obj["dimensions"]
        hw = float(dims[0]) * 0.5
        hd = float(dims[2]) * 0.5
        local_corners = np.array([
            [-hw, 0.0, -hd, 1.0],
            [+hw, 0.0, -hd, 1.0],
            [+hw, 0.0, +hd, 1.0],
            [-hw, 0.0, +hd, 1.0],
        ])
        world_corners = (m @ local_corners.T).T[:, :3]
        xs = list(world_corners[:, 0]) + [world_corners[0, 0]]
        ys = [-c for c in world_corners[:, 2]] + [-world_corners[0, 2]]
        ax.fill(xs, ys, facecolor='#2a2a4a', edgecolor='#888', linewidth=1, alpha=0.7)

    # ── 청취자 (xRIR 좌표 그대로, z=높이는 무시)
    lx = float(listener_pos["x"])
    ly = float(listener_pos["y"])
    ax.plot(lx, ly, 'o', color='#00d4ff', markersize=12, zorder=5)
    ax.annotate('Listener', (lx, ly), textcoords="offset points",
                xytext=(8, 8), color='#00d4ff', fontsize=8)

    # ── 임시 스피커
    if "initial" in speaker_positions:
        sp = speaker_positions["initial"]
        sx, sy = float(sp["x"]), float(sp["y"])
        ax.plot(sx, sy, 's', color='#ffd700', markersize=12, zorder=5)
        ax.annotate('Placeholder', (sx, sy), textcoords="offset points",
                    xytext=(8, 8), color='#ffd700', fontsize=8)
        ax.plot([lx, sx], [ly, sy], '--', color='#ffd700', alpha=0.5, linewidth=1)
        dist = np.sqrt((sx - lx)**2 + (sy - ly)**2)
        ax.annotate(f'{dist:.1f}m', ((lx+sx)/2, (ly+sy)/2),
                    color='#ffd700', fontsize=7, ha='center')

    # ── 최적 스피커
    for key, color, label in [("left", "#00ff88", "L"), ("right", "#ff6b6b", "R")]:
        if key in speaker_positions:
            sp = speaker_positions[key]
            sx, sy = float(sp["x"]), float(sp["y"])
            ax.plot(sx, sy, 's', color=color, markersize=12, zorder=5)
            ax.annotate(f'Speaker {label}', (sx, sy), textcoords="offset points",
                        xytext=(8, 8), color=color, fontsize=8)
            ax.plot([lx, sx], [ly, sy], '--', color=color, alpha=0.5, linewidth=1)
            dist = np.sqrt((sx - lx)**2 + (sy - ly)**2)
            ax.annotate(f'{dist:.1f}m', ((lx+sx)/2, (ly+sy)/2),
                        color=color, fontsize=7, ha='center')

    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.set_xlabel('x (m)', color='#888', fontsize=8)
    ax.set_ylabel('y (m)', color='#888', fontsize=8)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
