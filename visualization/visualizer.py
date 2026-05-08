"""
visualizer.py
=============
Person 4 — Visualization & UI
CET251 Maze Solver Project — Pygame Version

Single-window UI:
  Tab 1 : Algorithm Comparison  (BFS / DFS / A* side by side)
  Tab 2 : Risk Heatmap          (AI risk scores per cell)
"""

import math
import pygame

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = ( 18,  18,  35)
PANEL_BG    = ( 28,  28,  50)
WALL        = ( 30,  30,  30)
EMPTY       = (230, 230, 230)
TRAP        = (210,  40,  40)
START       = ( 50, 200,  80)
GOAL        = (255, 200,   0)
PATH        = ( 80, 160, 255)
ARROW       = (255, 255, 255)
TEXT_W      = (255, 255, 255)
TEXT_G      = (160, 210, 130)
ACCENT      = ( 91,  94, 166)
TAB_ACT     = ( 91,  94, 166)
TAB_IDLE    = ( 50,  50,  80)

RISK_COLORS = [
    (  0, 200,  80),
    (120, 210,  50),
    (200, 220,   0),
    (255, 180,   0),
    (255, 100,   0),
    (220,  30,  30),
]


def _lerp_color(t):
    t = max(0.0, min(1.0, t))
    idx = t * (len(RISK_COLORS) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(RISK_COLORS) - 1)
    f = idx - lo
    c0, c1 = RISK_COLORS[lo], RISK_COLORS[hi]
    return tuple(int(c0[i] + (c1[i] - c0[i]) * f) for i in range(3))


def _cell_color(cell, r, c, path_set):
    if cell == 1:           return WALL
    if cell == 'S':         return START
    if cell == 'G':         return GOAL
    if cell == 'T':         return TRAP
    if (r, c) in path_set:  return PATH
    return EMPTY


def _draw_maze(surface, maze, path, x0, y0, cell_px, title, font_sm, font_ti):
    rows, cols = len(maze), len(maze[0])
    path_set   = set(path) if path else set()

    t = font_ti.render(title, True, TEXT_W)
    surface.blit(t, (x0 + (cols * cell_px - t.get_width()) // 2, y0 - 28))

    for r in range(rows):
        for c in range(cols):
            cell  = maze[r][c]
            color = _cell_color(cell, r, c, path_set)
            rect  = pygame.Rect(x0 + c * cell_px, y0 + r * cell_px, cell_px, cell_px)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, PANEL_BG, rect, 1)
            if cell in ('S', 'G', 'T'):
                lbl = font_sm.render(str(cell), True,
                                     (0, 0, 0) if cell == 'G' else TEXT_W)
                surface.blit(lbl, lbl.get_rect(center=rect.center))

    if path and len(path) > 1:
        for i in range(len(path) - 1):
            r1, c1 = path[i];  r2, c2 = path[i + 1]
            cx1 = x0 + c1 * cell_px + cell_px // 2
            cy1 = y0 + r1 * cell_px + cell_px // 2
            cx2 = x0 + c2 * cell_px + cell_px // 2
            cy2 = y0 + r2 * cell_px + cell_px // 2
            pygame.draw.line(surface, ARROW, (cx1, cy1), (cx2, cy2), 2)
            angle = math.atan2(cy2 - cy1, cx2 - cx1)
            hs = cell_px * 0.28
            for da in (0.45, -0.45):
                ex = cx2 - hs * math.cos(angle - da)
                ey = cy2 - hs * math.sin(angle - da)
                pygame.draw.line(surface, ARROW, (cx2, cy2), (int(ex), int(ey)), 2)


def _draw_heatmap(surface, maze, risk_grid, x0, y0, cell_px, font_sm, font_ti):
    rows, cols = len(maze), len(maze[0])

    t = font_ti.render("AI Risk Heatmap  (0.0 safe → 1.0 danger)", True, TEXT_W)
    surface.blit(t, (x0 + (cols * cell_px - t.get_width()) // 2, y0 - 28))

    for r in range(rows):
        for c in range(cols):
            val  = risk_grid[r][c]
            cell = maze[r][c]
            rect = pygame.Rect(x0 + c * cell_px, y0 + r * cell_px, cell_px, cell_px)
            if cell == 1:       color = WALL
            elif val < 0:       color = (60, 60, 60)
            else:               color = _lerp_color(val)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, PANEL_BG, rect, 1)
            if cell in ('S', 'G', 'T'):
                lbl = font_sm.render(str(cell), True, (0, 0, 0))
                surface.blit(lbl, lbl.get_rect(center=rect.center))
            elif cell != 1 and val >= 0:
                txt = font_sm.render(f"{val:.1f}", True,
                                     TEXT_W if val > 0.5 else (0, 0, 0))
                surface.blit(txt, txt.get_rect(center=rect.center))


def _legend(surface, items, x, y, font):
    for i, (color, label) in enumerate(items):
        rx = x + i * 110
        pygame.draw.rect(surface, color, (rx, y, 18, 18))
        pygame.draw.rect(surface, ACCENT, (rx, y, 18, 18), 1)
        surface.blit(font.render(label, True, TEXT_W), (rx + 22, y + 1))


# ── Public API ────────────────────────────────────────────────────────────────

def visualize(maze, path, algorithm_name):
    visualize_comparison(maze, {algorithm_name: {"path": path, "nodes": 0, "time": 0}})


def visualize_comparison(maze, results, risk_grid=None):
    pygame.init()
    rows, cols = len(maze), len(maze[0])
    n_algo     = len(results)

    CELL   = max(28, min(60, 400 // max(rows, cols)))
    MARGIN = 60
    TAB_H  = 44
    LEG_H  = 40
    STAT_H = 52

    maze_w = cols * CELL
    maze_h = rows * CELL
    WIN_W  = max(820, MARGIN * (n_algo + 1) + maze_w * n_algo)
    WIN_H  = TAB_H + MARGIN + 30 + maze_h + STAT_H + LEG_H + MARGIN // 2

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Maze Solver AI — CET251")

    font_lg = pygame.font.SysFont("segoeui", 20, bold=True)
    font_md = pygame.font.SysFont("segoeui", 14, bold=True)
    font_sm = pygame.font.SysFont("segoeui", max(9, CELL // 3), bold=True)
    font_ti = pygame.font.SysFont("segoeui", 13, bold=True)

    tabs    = ["Algorithm Comparison", "Risk Heatmap"]
    tab_idx = 0
    clock   = pygame.time.Clock()

    def draw_tab_bar(active):
        tw = WIN_W // len(tabs)
        for i, name in enumerate(tabs):
            color = TAB_ACT if i == active else TAB_IDLE
            pygame.draw.rect(screen, color, (i * tw, 0, tw, TAB_H))
            lbl = font_md.render(name, True, TEXT_W)
            screen.blit(lbl, lbl.get_rect(center=(i * tw + tw // 2, TAB_H // 2)))

    def draw_comparison():
        screen.fill(BG)
        draw_tab_bar(0)
        algo_names = list(results.keys())
        for idx, name in enumerate(algo_names):
            res   = results[name]
            path  = res.get("path") or []
            nodes = res.get("nodes", 0)
            t     = res.get("time",  0.0)
            traps = res.get("traps_hit", 0)
            x0 = MARGIN + idx * (maze_w + MARGIN)
            y0 = TAB_H + MARGIN + 30
            _draw_maze(screen, maze, path, x0, y0, CELL, name, font_sm, font_ti)
            sy = y0 + maze_h + 8
            stats = [f"Path : {len(path)} steps",
                     f"Nodes: {nodes}",
                     f"Time : {t:.5f}s",
                     f"Traps: {traps}"]
            for si, s in enumerate(stats):
                col = TRAP if (si == 3 and traps > 0) else TEXT_G
                lbl = font_ti.render(s, True, col)
                screen.blit(lbl, (x0 + (maze_w - lbl.get_width()) // 2,
                                  sy + si * 13))
        leg_items = [(EMPTY,"Empty"),(WALL,"Wall"),(TRAP,"Trap"),
                     (START,"Start"),(GOAL,"Goal"),(PATH,"Path")]
        total_w = len(leg_items) * 110
        _legend(screen, leg_items, (WIN_W - total_w) // 2, WIN_H - LEG_H - 4, font_ti)

    def draw_heatmap():
        screen.fill(BG)
        draw_tab_bar(1)
        if risk_grid is None:
            msg = font_lg.render("No risk data available", True, TEXT_W)
            screen.blit(msg, msg.get_rect(center=(WIN_W // 2, WIN_H // 2)))
            return
        x0 = (WIN_W - maze_w) // 2
        y0 = TAB_H + MARGIN + 30
        _draw_heatmap(screen, maze, risk_grid, x0, y0, CELL, font_sm, font_ti)
        bar_w, bar_h = 200, 16
        bx = (WIN_W - bar_w) // 2
        by = WIN_H - LEG_H + 4
        for i in range(bar_w):
            pygame.draw.line(screen, _lerp_color(i / bar_w),
                             (bx + i, by), (bx + i, by + bar_h))
        pygame.draw.rect(screen, ACCENT, (bx, by, bar_w, bar_h), 1)
        screen.blit(font_ti.render("Safe (0.0)", True, TEXT_G), (bx - 68, by + 1))
        screen.blit(font_ti.render("Danger (1.0)", True, TRAP),  (bx + bar_w + 4, by + 1))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if my < TAB_H:
                    tw = WIN_W // len(tabs)
                    tab_idx = mx // tw

        if tab_idx == 0:
            draw_comparison()
        else:
            draw_heatmap()

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def visualize_risk_heatmap(maze, risk_grid, algorithm_name=""):
    pass
