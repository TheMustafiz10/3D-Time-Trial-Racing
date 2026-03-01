


from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
import time
import random

try:
    from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_18 as GLUT_BITMAP_HELVETICA_18
except Exception:
    try:
        from OpenGL import GLUT as _gmod
        GLUT_BITMAP_HELVETICA_18 = getattr(_gmod, 'GLUT_BITMAP_HELVETICA_18', getattr(_gmod, 'GLUT_BITMAP_9_BY_15', 0))
    except Exception:
        GLUT_BITMAP_HELVETICA_18 = 0

game_state = 'menu'
menu_track_selection = 0
menu_weather_selection = 0
menu_difficulty_selection = 1
menu_lap_selection = 0
menu_time_selection = 0
menu_current_option = 0
game_over_message = ""

car_x, car_y, car_z = 0, 0.5, -5
car_angle = 0.0
car_speed = 0.0
max_speed_base = 1.25
boosted_max_speed = 1.50
nos_max_speed = 1.80
acceleration = 0.012
brake_power = -0.015
steering_angle = 2.5
friction = 0.99

camera_mode = 0
camera_distance = 10.0
camera_height = 5.0

lap_start_time = 0
current_lap_time = 0
best_times = {0: float('inf'), 1: float('inf'), 2: float('inf'), 3: float('inf')}
lap_count = 0
on_lap = False
target_laps = float('inf')

time_limit = float('inf')
game_start_time = 0
time_limit_options = [0, 30, 60, 90, 120, 180]
final_time_left = None

health = 3
collision_cooldown = 0

boost_active = False
boost_timer = 0
boost_duration = 2.0
boost_multiplier = 1.20

nos_count = 0
nos_max = 3
nos_active = False
nos_timer = 0.0
nos_duration = 2.0
nos_multiplier = 1.50
nos_pickups = []
inactive_nos_pickups = []
nos_respawn_delay = 20.0

track_selection = 0
weather_mode = 0
weather_names = {0: "Snow", 1: "Rain", 2: "Sunny"}
difficulty_level = 1
difficulty_names = {0: "Easy", 1: "Medium", 2: "Hard"}

day_color = [0.5, 0.8, 1.0, 1.0]
snow_sky = [0.82, 0.87, 0.92, 1.0]
rain_sky = [0.45, 0.52, 0.60, 1.0]

track_width = 10

FENCE_GAP = track_width * 0.8
FENCE_HEIGHT = 1.4
FENCE_POST_HALF = 0.08
FENCE_POST_STEP = 1
FENCE_CLAMP_CLEAR = 0.25
FENCE_TOUCH_SLOW = 0.80
FENCE_SCRAPE_SLOW = 0.98
FENCE_POST_INSET = 0.12

WAYPOINTS_BASE = [
    (-80, -40), ( -20, -40), ( 40, -55), ( 85, -10),
    ( 65,  40), (   5,  65), (-60,  45), (-85,   0),
]
WAYPOINTS_ALT = [
    (0,     0), (140,   0), (260, 140), (300, 300), (220, 430), (60,  520),
    (-140, 500), (-240, 360), (-260, 180), (-200,  40), (-80, -40), (-20, -60),
    (0,   -40)
]
WAYPOINTS_T3 = [
    (-150, -100), (150, -100), (200, -80), (200, 80), (150, 100),
    (-150, 100), (-200, 80), (-200, -80),
]
WAYPOINTS_T4 = [
    (0, -120), (180, -120), (250, -50), (200, 100), (50, 150),
    (-150, 100), (-220, 0), (-150, -100),
]

current_waypoints = WAYPOINTS_BASE

track_samples = []
START_LINE_P0 = None
START_LINE_P1 = None
prev_start_side = None

START_LINE_OFFSET_BY_TRACK = {0: 20, 1: 24}

def _get_start_line_index() -> int:
    """Index in samples used to place the start line for the current track."""
    try:
        off = START_LINE_OFFSET_BY_TRACK.get(track_selection, 0)
    except Exception:
        off = 0
    if track_samples:
        return off % max(1, (len(track_samples) - 1))
    return 0

def _build_track_geometry():
    """Sample waypoints and build smoothed track samples and start line."""
    global track_samples, START_LINE_P0, START_LINE_P1, prev_start_side
    wp = current_waypoints
    n = len(wp)
    dists = [0.0]
    total = 0.0
    for i in range(n):
        x0,z0 = wp[i]
        x1,z1 = wp[(i+1)%n]
        seg_len = math.hypot(x1-x0, z1-z0)
        total += seg_len
        dists.append(total)

    step = 2.0
    
    raw_points = []
    s = 0.0
    while s < total:
        for i in range(n):
            if dists[i] <= s < dists[i+1]:
                break
        x0,z0 = wp[i]
        x1,z1 = wp[(i+1)%n]
        seg_len = dists[i+1]-dists[i]
        t = (s - dists[i]) / seg_len if seg_len > 0 else 0
        cx = x0 + (x1-x0)*t
        cz = z0 + (z1-z0)*t
        raw_points.append((cx, cz))
        s += step

    if not raw_points:
        return

    num_points = len(raw_points)

    initial_tangents = []
    for i in range(num_points):
        prev_point = raw_points[(i - 1 + num_points) % num_points]
        next_point = raw_points[(i + 1) % num_points]
        tx = next_point[0] - prev_point[0]
        tz = next_point[1] - prev_point[1]
        mag = math.hypot(tx, tz)
        if mag > 0:
            initial_tangents.append((tx / mag, tz / mag))
        else:
            initial_tangents.append((1.0, 0.0))

    smoothed_tangents = []
    window_size = 3 
    for i in range(num_points):
        avg_tx, avg_tz = 0.0, 0.0
        for j in range(-window_size, window_size + 1):
            sample_index = (i + j + num_points) % num_points
            tx, tz = initial_tangents[sample_index]
            avg_tx += tx
            avg_tz += tz
        
        mag = math.hypot(avg_tx, avg_tz)
        if mag > 0:
            smoothed_tangents.append((avg_tx / mag, avg_tz / mag))
        else:
            smoothed_tangents.append((1.0, 0.0))

    track_samples = []
    for i in range(num_points):
        cx, cz = raw_points[i]
        tx, tz = smoothed_tangents[i]
        nx, nz = -tz, tx
        track_samples.append((cx, cz, tx, tz, nx, nz))


    if track_samples:
        fcx,fcz,ftx,ftz,fnx,fnz = track_samples[0]
        track_samples.append((fcx,fcz,ftx,ftz,fnx,fnz))
        sidx = _get_start_line_index()
        x0,z0,tx,tz,nx,nz = track_samples[sidx]
        w2 = track_width*0.5
        START_LINE_P0 = (x0 + nx*w2, z0 + nz*w2)
        START_LINE_P1 = (x0 - nx*w2, z0 - nz*w2)
        prev_start_side = None

def init_car_position():
    """Place the car at the start sample and align to track tangent."""
    global car_x, car_z, car_angle
    if track_samples:
        sidx = _get_start_line_index()
        cx, cz, tx, tz, _, _ = track_samples[sidx]
        car_x, car_z = cx, cz
        car_angle = math.degrees(math.atan2(tx, tz))
        return
    if len(current_waypoints) < 2:
        return
    x0, z0 = current_waypoints[0]
    x1, z1 = current_waypoints[1]
    car_x, car_z = x0, z0
    car_angle = math.degrees(math.atan2(x1 - x0, z1 - z0))

_build_track_geometry()
init_car_position()

precip_particles = []
precip_last_time = 0.0

def _setup_precipitation():
    """Initialize precipitation particles for current weather."""
    global precip_particles, precip_last_time
    precip_particles = []
    precip_last_time = time.time()
    if weather_mode not in (0, 1):
        return
    if weather_mode == 0:
        count = 250
        wind_speed = 20.0
        for _ in range(count):
            layer = random.uniform(0.2, 1.0)
            x = random.uniform(-10.0, 810.0)
            y = random.uniform(-10.0, 610.0)
            size = 0.5 + layer * 2.0
            vx = wind_speed * layer
            vy = (20.0 + layer * 80.0)
            precip_particles.append([x, y, layer, vx, vy, size])
    else:
        count = 450
        top_min, top_max = 40.0, 70.0
        for _ in range(count):
            x = car_x + (random.random() - 0.5) * 100.0
            z = car_z + (random.random() - 0.5) * 100.0
            y = random.uniform(top_min, top_max)
            vy = 28.0
            precip_particles.append([x, y, z, vy])



def _update_precipitation(dt: float):
    """Advance precipitation and respawn out-of-bounds particles."""
    if not precip_particles:
        return
    if weather_mode == 0:
        for p in precip_particles:
            p[0] += p[3] * dt
            p[1] -= p[4] * dt
            if p[1] < -p[5]:
                p[1] = 600.0 + p[5]
                p[0] = random.uniform(0.0, 800.0)
            if p[0] > 800.0 + p[5]:
                p[0] = -p[5]
                p[1] = random.uniform(0.0, 600.0)
    else:
        top_min, top_max = 40.0, 70.0
        for p in precip_particles:
            p[1] -= p[3] * dt
            if p[1] < 0.0:
                p[0] = car_x + (random.random() - 0.5) * 100.0
                p[2] = car_z + (random.random() - 0.5) * 100.0
                p[1] = random.uniform(top_min, top_max)



def _draw_precipitation():
    """Draw precipitation."""
    if not precip_particles:
        return
    if weather_mode == 1:
        glBegin(GL_QUADS)
        glColor3f(0.7, 0.8, 1.0)
        half_w = 0.06
        h = 1.6
        for x, y, z, vy in precip_particles:
            glVertex3f(x - half_w, y, z)
            glVertex3f(x + half_w, y, z)
            glVertex3f(x + half_w, y - h, z)
            glVertex3f(x - half_w, y - h, z)
        glEnd()
    elif weather_mode == 0:
        def draw_snow_overlay():
            glBegin(GL_QUADS)
            for x, y, layer, _, _, size in precip_particles:
                shade = 0.7 + 0.3 * layer
                glColor3f(shade, shade, shade)
                half = size / 2.0
                glVertex3f(x - half, y + half, 0.0)
                glVertex3f(x + half, y + half, 0.0)
                glVertex3f(x + half, y - half, 0.0)
                glVertex3f(x - half, y - half, 0.0)
            glEnd()
        draw_2d_scene(draw_snow_overlay)

def _sample_at_fraction(frac):
    """Return track sample at the given fractional progress (0..1)."""
    if not track_samples: return (0,0,1,0,0,1)
    idx = int(frac * (len(track_samples)-1)) % len(track_samples)
    return track_samples[idx]

def _nearest_track_frame(px: float, pz: float):
    """Nearest track frame (center, tangent, normal) to a given point."""
    if not track_samples:
        return (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    best = None
    best_d2 = 1e18
    stride = 6
    for i in range(0, len(track_samples)-1, stride):
        cx, cz, tx, tz, nx, nz = track_samples[i]
        d2 = (px - cx)*(px - cx) + (pz - cz)*(pz - cz)
        if d2 < best_d2:
            best_d2 = d2
            best = (i, cx, cz, tx, tz, nx, nz)
    i0 = max(0, best[0] - stride)
    i1 = min(len(track_samples)-1, best[0] + stride)
    for i in range(i0, i1):
        cx, cz, tx, tz, nx, nz = track_samples[i]
        d2 = (px - cx)*(px - cx) + (pz - cz)*(pz - cz)
        if d2 < best_d2:
            best_d2 = d2
            best = (i, cx, cz, tx, tz, nx, nz)
    _, cx, cz, tx, tz, nx, nz = best
    return (cx, cz, tx, tz, nx, nz)



def _lateral_offset_from_center(px: float, pz: float):
    """Signed lateral offset from center; positive is left of the tangent."""
    cx, cz, tx, tz, nx, nz = _nearest_track_frame(px, pz)
    vx, vz = (px - cx), (pz - cz)
    lat = vx*nx + vz*nz
    return lat, (cx, cz, tx, tz, nx, nz)

def _build_obstacles_and_boosts():
    """Compute obstacle and boost pad positions by track and difficulty."""
    obs_fracs_easy = [
        [0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
        [0.1 + x * (0.9 / 21) for x in range(21)],  # Track 2 (21 obs, 1.5x of 14)
        [0.15 + x * (0.85 / 18) for x in range(18)], # Track 3 (18 obs, 1.5x of 12)
        [0.15 + x * (0.85 / 14) for x in range(14)]  # Track 4 (14 obs, 1.4x of 10)
    ]
    obs_fracs_medium = [
        [0.10, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91],
        [0.1 + x * (0.9 / 33) for x in range(33)],  # Track 2 (33 obs, 1.5x of 22)
        [0.45 + x * (0.85 / 27) for x in range(27)], # Track 3 (27 obs, 1.5x of 18)
        [0.15 + x * (0.85 / 22) for x in range(22)]  # Track 4 (22 obs, 1.4x of 16)
    ]
    obs_fracs_hard = [
        [0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.96],
        [0.1 + x * (0.9 / 67) for x in range(67)],  # Track 2 (67 obs, 1.5x of 45)
        [0.15 + x * (0.85 / 46) for x in range(46)], # Track 3 (46 obs, 1.5x of 31)
        [0.15 + x * (0.85 / 45) for x in range(45)]  # Track 4 (45 obs, 1.4x of 32)
    ]

    if difficulty_level == 0:
        obs_fracs = obs_fracs_easy
    elif difficulty_level == 2:
        obs_fracs = obs_fracs_hard
    else:
        obs_fracs = obs_fracs_medium

    boost_fracs = [ 
        [0.15, 0.45, 0.65, 0.85], 
        [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95],
        [0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
        [0.1 + x * (0.85 / 18) for x in range(18)]
    ]
    all_obs, all_boost = [], []
    for fracs in obs_fracs:
        lst = []
        for i, f in enumerate(fracs):
            cx, cz, _, _, nx, nz = _sample_at_fraction(f)
            side = 1 if i % 2 == 0 else -1
            lst.append((cx + nx * side * track_width*0.3, 0.5, cz + nz * side * track_width*0.3))
        all_obs.append(lst)
    for fracs in boost_fracs:
        lst = []
        for f in fracs:
            cx, cz, _, _, _, _ = _sample_at_fraction(f)
            lst.append((cx, 0.1, cz))
        all_boost.append(lst)
    return all_obs, all_boost

all_obstacles, all_boost_pads = _build_obstacles_and_boosts()

def _build_nos_pickups():
    nos_fracs = [ 
        [0.25, 0.50, 0.75, 0.95],
        [0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
        [0.18, 0.36, 0.54, 0.72, 0.90],
        [x * (1/12) for x in range(1, 13)]
    ]
    all_nos = []
    for fracs in nos_fracs:
        lst = []
        for f in fracs:
            cx, cz, _, _, _, _ = _sample_at_fraction(f)
            lst.append((cx, 1.2, cz))
        all_nos.append(lst)
    return all_nos

nos_pickups = _build_nos_pickups()

keys = { 'w': False, 's': False, 'a': False, 'd': False, ' ': False }

def init():
    """Initialize OpenGL state."""
    glEnable(GL_DEPTH_TEST)

def draw_text_colored(x, y, text, color=(1.0, 1.0, 1.0)):
    """Draw 2D text at screen coordinates with a color."""
    glColor3f(color[0], color[1], color[2])
    glRasterPos2f(x, y)
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

def _text_pixel_width(text: str) -> int:
    """Approximate pixel width for GLUT bitmap text."""
    return int(len(text) * 9)

def draw_2d_scene(draw_func):
    """Run a draw callback in a temporary 2D orthographic projection."""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, 800, 0, 600)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    draw_func()

    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()

def _fill_background_rgb(r, g, b):
    """Fill the screen with a solid color (2D) and clear depth for 3D."""
    def draw_bg():
        glBegin(GL_QUADS)
        glColor3f(r, g, b)
        glVertex3f(0,   0,   0.0)
        glVertex3f(800, 0,   0.0)
        glVertex3f(800, 600, 0.0)
        glVertex3f(0,   600, 0.0)
        glEnd()
    draw_2d_scene(draw_bg)
    glClear(GL_DEPTH_BUFFER_BIT)

def draw_floor():
    """Draw a tiled ground plane around the car."""
    visible_range, tile_size = 250, 5
    snapped_x = int(car_x / tile_size) * tile_size
    snapped_z = int(car_z / tile_size) * tile_size
    glBegin(GL_QUADS)
    for i in range(snapped_x - visible_range, snapped_x + visible_range, tile_size):
        for j in range(snapped_z - visible_range, snapped_z + visible_range, tile_size):
            c = (0.20, 0.75, 0.22) if ((i//tile_size + j//tile_size) & 1) == 0 else (0.16, 0.65, 0.20)
            glColor3f(*c)
            glVertex3f(i, 0, j); glVertex3f(i + tile_size, 0, j)
            glVertex3f(i + tile_size, 0, j + tile_size); glVertex3f(i, 0, j + tile_size)
    glEnd()

def draw_track():
    """Draw the road surface along the sampled centerline."""
    if not track_samples: return
    w2 = track_width*0.5
    glBegin(GL_QUADS)
    for i in range(len(track_samples)-1):
        cx0, cz0, _, _, nx0, nz0 = track_samples[i]
        cx1, cz1, _, _, nx1, nz1 = track_samples[i+1]
        c = (0.26,0.26,0.28) if (i//6) % 2 == 0 else (0.22,0.22,0.24)
        glColor3f(*c)
        glVertex3f(cx0+nx0*w2, 0.1, cz0+nz0*w2); glVertex3f(cx1+nx1*w2, 0.1, cz1+nz1*w2)
        glVertex3f(cx1-nx1*w2, 0.1, cz1-nz1*w2); glVertex3f(cx0-nx0*w2, 0.1, cz0-nz0*w2)
    glEnd()

def draw_fences():
    """Draw fence posts and rails along both sides of the track."""
    if not track_samples: return
    w2 = track_width*0.5
    gap = FENCE_GAP
    post_h = FENCE_HEIGHT
    s = FENCE_POST_HALF
    inset = min(FENCE_POST_INSET, max(0.0, gap * 0.5))
    glColor3f(0.6, 0.45, 0.25)
    for i in range(0, len(track_samples)-1, FENCE_POST_STEP):
        cx, cz, _, _, nx, nz = track_samples[i]
        lx, lz = cx + nx*(w2 + gap - inset), cz + nz*(w2 + gap - inset)
        rx, rz = cx - nx*(w2 + gap - inset), cz - nz*(w2 + gap - inset)
        glPushMatrix(); glTranslatef(lx, 0.0, lz); glScalef(s*2, post_h, s*2); glutSolidCube(1.0); glPopMatrix()
        glPushMatrix(); glTranslatef(rx, 0.0, rz); glScalef(s*2, post_h, s*2); glutSolidCube(1.0); glPopMatrix()

    rail_y0, rail_y1 = 0.35, 0.55
    glColor3f(0.62, 0.48, 0.28)
    glBegin(GL_QUADS)
    for i in range(0, len(track_samples)-1):
        cx0, cz0, _, _, nx0, nz0 = track_samples[i]
        cx1, cz1, _, _, nx1, nz1 = track_samples[i+1]
        lx0, lz0 = cx0 + nx0*(w2 + gap), cz0 + nz0*(w2 + gap)
        lx1, lz1 = cx1 + nx1*(w2 + gap), cz1 + nz1*(w2 + gap)
        glVertex3f(lx0, rail_y0, lz0)
        glVertex3f(lx1, rail_y0, lz1)
        glVertex3f(lx1, rail_y1, lz1)
        glVertex3f(lx0, rail_y1, lz0)
        rx0, rz0 = cx0 - nx0*(w2 + gap), cz0 - nz0*(w2 + gap)
        rx1, rz1 = cx1 - nx1*(w2 + gap), cz1 - nz1*(w2 + gap)
        glVertex3f(rx0, rail_y0, rz0)
        glVertex3f(rx1, rail_y0, rz1)
        glVertex3f(rx1, rail_y1, rz1)
        glVertex3f(rx0, rail_y1, rz0)
    glEnd()

def draw_start_finish_line():
    """Draw the start/finish line quad across the road."""
    if START_LINE_P0 is None or not track_samples: return
    (lx, lz) = START_LINE_P0
    (rx, rz) = START_LINE_P1
    thickness = 2.0
    dx, dz = (rx - lx), (rz - lz)
    length = math.hypot(dx, dz)
    if length == 0:
        tx, tz = 0.0, 1.0
    else:
        tx, tz = (dz / length), (-dx / length)
    lx_b, lz_b = lx - tx * thickness * 0.5, lz - tz * thickness * 0.5
    lx_f, lz_f = lx + tx * thickness * 0.5, lz + tz * thickness * 0.5
    rx_f, rz_f = rx + tx * thickness * 0.5, rz + tz * thickness * 0.5
    rx_b, rz_b = rx - tx * thickness * 0.5, rz - tz * thickness * 0.5
    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glVertex3f(lx_b, 0.15, lz_b)
    glVertex3f(lx_f, 0.15, lz_f)
    glVertex3f(rx_f, 0.15, rz_f)
    glVertex3f(rx_b, 0.15, rz_b)
    glEnd()

def draw_tree_at(x, z, scale=1.0):
    """Draw a simple tree model at world position."""
    glPushMatrix()
    glTranslatef(x, 0, z)
    glRotatef(-90, 1, 0, 0)
    glColor3f(0.45, 0.30, 0.10)
    gluCylinder(gluNewQuadric(), 0.5 * scale, 0.5 * scale, 4.0 * scale, 12, 1)
    glTranslatef(0, 0, 4.0 * scale)
    glColor3f(0.05, 0.55, 0.12)
    gluSphere(gluNewQuadric(), 2.2 * scale, 14, 12)
    glTranslatef(0.8 * scale, -0.6 * scale, 0.8 * scale)
    glColor3f(0.04, 0.48, 0.10)
    gluSphere(gluNewQuadric(), 1.6 * scale, 12, 10)
    glPopMatrix()

def draw_scenery_and_items():
    """Draw trees, obstacles, boost pads, and NOS pickups."""
    if track_samples:
        margin = track_width*0.5 + 14.0
        if track_selection == 0:
            margin += 8.0
        for i in range(0, len(track_samples), 24):
            cx,cz,_,_,nx,nz = track_samples[i]
            lx, lz = cx + nx * margin, cz + nz * margin
            rx, rz = cx - nx * margin, cz - nz * margin
            if (i // 24) % 2 == 0:
                draw_tree_at(lx, lz, 1.2)
                draw_tree_at(rx, rz, 1.0)
            else:
                draw_tree_at(lx, lz, 0.9)
   
    
    if difficulty_level == 0: obstacle_size = 2.0
    elif difficulty_level == 1: obstacle_size = 2.4
    else: obstacle_size = 2.8

    glColor3f(0.8,0.8,0.2)
    for (x,y,z) in all_obstacles[track_selection]:
        glPushMatrix(); glTranslatef(x,y,z); glutSolidCube(obstacle_size); glPopMatrix()

 
    
    glColor3f(0.2,0.8,0.8)
    for (x,y,z) in all_boost_pads[track_selection]:
        glPushMatrix(); glTranslatef(x,y,z); glScalef(4,0.1,4); glutSolidCube(1); glPopMatrix()

  
    
    
    glColor3f(0.9,0.35,0.1); t = time.time()
    for (x,y,z) in nos_pickups[track_selection]:
        glPushMatrix()
        glTranslatef(x, y + math.sin(t*4)*0.2, z)
        glRotatef((t*120)%360, 0,1,0)
        gluSphere(gluNewQuadric(), 0.6, 14, 12)
        glPopMatrix()
        
        

def draw_car():
    """Draw the player car and active NOS flame."""
    glPushMatrix(); glTranslatef(car_x, car_y, car_z); glRotatef(car_angle, 0, 1, 0)

    
    glPushMatrix(); glColor3f(1, 0, 0); glScalef(1.5, 0.5, 3.0); glutSolidCube(1); glPopMatrix()

    
    glPushMatrix(); glColor3f(0.8, 0.8, 1); glTranslatef(0, 0.5, 0); glScalef(1.2, 0.4, 1.5); glutSolidCube(1); glPopMatrix()

    
    glColor3f(0.1, 0.1, 0.1)
    for p in [(-0.8,-0.2,1.0), (0.8,-0.2,1.0), (-0.8,-0.2,-1.0), (0.8,-0.2,-1.0)]:
        glPushMatrix(); glTranslatef(*p); glutSolidCube(0.4); glPopMatrix()
    if nos_active:
        glPushMatrix()
        glTranslatef(0, 0.1, -1.9)
        p = 0.7 + 0.3*math.sin(time.time()*25)
        glScalef(0.4*p, 0.4*p, 1.3*p)
        glColor3f(1.0,0.5,0.1)
        gluCylinder(gluNewQuadric(), 0.8, 0.0, 2.0, 12, 6)
        glPopMatrix()
    glPopMatrix()



def update_camera():
    """Set the camera (third- or first-person)."""
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(60, (800 / 600), 0.1, 2000.0)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    if camera_mode == 0:
        cam_x = car_x - camera_distance * math.sin(math.radians(car_angle))
        cam_z = car_z - camera_distance * math.cos(math.radians(car_angle))
        gluLookAt(cam_x, car_y + camera_height, cam_z, car_x, car_y, car_z, 0, 1, 0)
    else:
        cam_x = car_x + 1.5 * math.sin(math.radians(car_angle))
        cam_z = car_z + 1.5 * math.cos(math.radians(car_angle))
        look_at_x = cam_x + 10 * math.sin(math.radians(car_angle))
        look_at_z = cam_z + 10 * math.cos(math.radians(car_angle))
        gluLookAt(cam_x, car_y + 1.0, cam_z, look_at_x, car_y, look_at_z, 0, 1, 0)

def draw_main_menu():
    """Draw the main menu and highlight the selected item."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    _fill_background_rgb(0.1, 0.1, 0.2)

    def menu_text():
        draw_text_colored(280, 520, "3D Time Trial Racing", (1, 1, 0))
        
        best_time = best_times.get(menu_track_selection, float('inf'))
        time_str = f"{best_time:.2f}" if best_time != float('inf') else "N/A"
        track_text = f"Track {menu_track_selection + 1} < Best: {time_str} >"

        weather_text = f"Weather     < {weather_names[menu_weather_selection]} >"
        difficulty_text = f"Difficulty  < {difficulty_names[menu_difficulty_selection]} >"
        lap_text = f"Laps        < {menu_lap_selection if menu_lap_selection > 0 else 'Free'} >"
        
        time_val = time_limit_options[menu_time_selection]
        time_text = f"Time Limit  < {str(time_val) + 's' if time_val > 0 else 'Free'} >"

        start_text = "Start Game"

        draw_text_colored(260, 420, track_text, (1, 0.8, 0) if menu_current_option == 0 else (1, 1, 1))
        draw_text_colored(260, 380, weather_text, (1, 0.8, 0) if menu_current_option == 1 else (1, 1, 1))
        draw_text_colored(260, 340, difficulty_text, (1, 0.8, 0) if menu_current_option == 2 else (1, 1, 1))
        draw_text_colored(260, 300, lap_text, (1, 0.8, 0) if menu_current_option == 3 else (1, 1, 1))
        draw_text_colored(260, 260, time_text, (1, 0.8, 0) if menu_current_option == 4 else (1, 1, 1))
        draw_text_colored(340, 180, start_text, (0, 1, 0) if menu_current_option == 5 else (1, 1, 1))
        
        draw_text_colored(250, 80, "Use W/S to navigate, A/D to change.", (0.8, 0.8, 0.8))
        draw_text_colored(310, 50, "Press Enter to start.", (0.8, 0.8, 0.8))
    draw_2d_scene(menu_text)

def draw_game_world():
    """Draw the 3D world, precipitation, and HUD."""
    global precip_last_time
    if weather_mode == 0:     
        bg = snow_sky
    elif weather_mode == 1:   
        bg = rain_sky
    else:                     
        bg = day_color

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    _fill_background_rgb(bg[0], bg[1], bg[2])

    update_camera()
    now = time.time()
    dt = max(0.0, now - precip_last_time) if precip_last_time else 0.0
    precip_last_time = now
    _update_precipitation(dt)
    draw_floor()
    draw_track()
    draw_fences()
    draw_start_finish_line()
    draw_scenery_and_items()
    draw_car()
    _draw_precipitation()
    draw_hud()

def display():
    """Main display callback; draws menu or game."""
    if game_state == 'menu':
        draw_main_menu()
    elif game_state in ['playing', 'game_over']:
        draw_game_world()
    glutSwapBuffers()

def draw_hud():
    """Draw the HUD (lap, time, speed, health, NOS)."""
    def hud_text():
        if target_laps == float('inf'):
            draw_text_colored(10, 580, f"Lap: {lap_count}")
        else:
            draw_text_colored(10, 580, f"Lap: {lap_count} / {target_laps}")

        draw_text_colored(10, 560, f"Time: {current_lap_time:.2f}")
        current_best = best_times.get(track_selection, float('inf'))
        draw_text_colored(10, 540, f"Best: {current_best:.2f}" if current_best != float('inf') else "Best: N/A")
        
        if time_limit != float('inf'):
            if game_state == 'game_over' and final_time_left is not None:
                remaining_time = final_time_left
            elif game_start_time > 0:
                remaining_time = max(0, time_limit - (time.time() - game_start_time))
            else:
                remaining_time = time_limit
            draw_text_colored(350, 580, f"Time Left: {int(remaining_time // 60)}:{int(remaining_time % 60):02d}")

        draw_text_colored(650, 580, f"Speed: {abs(car_speed * 100):.0f} km/h")
        draw_text_colored(650, 560, f"Boost: {'ON' if boost_active else 'OFF'}")
        draw_text_colored(650, 540, f"Health: {health}")
        draw_text_colored(650, 520, f"NOS: {nos_count}/{nos_max}")
        
        if game_state == 'game_over':
            msg = game_over_message
            hint = "Press 'R' or ESC for menu"
            x_msg = (800 - _text_pixel_width(msg)) / 2
            x_hint = (800 - _text_pixel_width(hint)) / 2
            y_center = 300
            draw_text_colored(x_msg, y_center, msg, (1,0.2,0.2))
            draw_text_colored(x_hint, y_center - 24, hint)
    draw_2d_scene(hud_text)



def update_game_logic():
    """Advance game state: input, physics, collisions, laps, boosts, NOS."""
    global car_speed, car_x, car_z, car_angle, collision_cooldown, health, on_lap, lap_start_time, current_lap_time, lap_count, prev_start_side
    global boost_active, boost_timer, nos_active, nos_timer, nos_count, game_state, inactive_nos_pickups, game_over_message, game_start_time, final_time_left

    if time_limit != float('inf'):
        if game_start_time > 0 and time.time() - game_start_time >= time_limit:
            if game_start_time > 0:
                now_t = time.time()
                current_lap_time = now_t - game_start_time
                final_time_left = 0
            game_state = 'game_over'
            if target_laps != float('inf') and lap_count < target_laps:
                game_over_message = "You Lost! Time Finished."
            else:
                game_over_message = "Time's Up!"
            return

    if keys['w'] and game_start_time == 0:
        game_start_time = time.time()
        
    lat, frame = _lateral_offset_from_center(car_x, car_z)
    if keys['w']: car_speed += acceleration
    if keys['s']: car_speed += brake_power
    road_half = track_width * 0.5
    is_off_track = abs(lat) > road_half
    off_track_max_speed = 1.15

    if is_off_track:
        cap = off_track_max_speed
    else:
        if nos_active:
            cap = nos_max_speed
        elif boost_active:
            cap = boosted_max_speed
        else:
            cap = max_speed_base

    car_speed = max(-cap / 2, min(cap, car_speed)) * friction
    if abs(car_speed) > 0.01:
        if keys['a']: car_angle += steering_angle * (car_speed / max_speed_base)
        if keys['d']: car_angle -= steering_angle * (car_speed / max_speed_base)
    car_x += car_speed * math.sin(math.radians(car_angle))
    car_z += car_speed * math.cos(math.radians(car_angle))

   
    
    if track_samples:
        cx, cz, tx, tz, nx, nz = frame
        fence_lat = road_half + FENCE_GAP
        inside_limit = fence_lat - FENCE_CLAMP_CLEAR
        if lat > fence_lat:
            lat_clamped = inside_limit
            car_x = cx + nx * lat_clamped
            car_z = cz + nz * lat_clamped
            hx, hz = math.sin(math.radians(car_angle)), math.cos(math.radians(car_angle))
            tsgn = 1.0 if (hx*tx + hz*tz) >= 0.0 else -1.0
            car_angle = math.degrees(math.atan2(tsgn*tx, tsgn*tz))
            car_speed *= FENCE_TOUCH_SLOW
            if car_speed < 0.05:
                car_speed = 0.05
        elif lat < -fence_lat:
            lat_clamped = -inside_limit
            car_x = cx + nx * lat_clamped
            car_z = cz + nz * lat_clamped
            hx, hz = math.sin(math.radians(car_angle)), math.cos(math.radians(car_angle))
            tsgn = 1.0 if (hx*tx + hz*tz) >= 0.0 else -1.0
            car_angle = math.degrees(math.atan2(tsgn*tx, tsgn*tz))
            car_speed *= FENCE_TOUCH_SLOW
            if car_speed < 0.05:
                car_speed = 0.05
        else:
            band = 0.8
            if lat > inside_limit - band:
                car_speed *= FENCE_SCRAPE_SLOW
                hx, hz = math.sin(math.radians(car_angle)), math.cos(math.radians(car_angle))
                tsgn = 1.0 if (hx*tx + hz*tz) >= 0.0 else -1.0
                car_x += 0.02 * tsgn * tx
                car_z += 0.02 * tsgn * tz
            elif lat < -(inside_limit - band):
                car_speed *= FENCE_SCRAPE_SLOW
                hx, hz = math.sin(math.radians(car_angle)), math.cos(math.radians(car_angle))
                tsgn = 1.0 if (hx*tx + hz*tz) >= 0.0 else -1.0
                car_x += 0.02 * tsgn * tx
                car_z += 0.02 * tsgn * tz

    if collision_cooldown > 0: collision_cooldown -= 1
    if collision_cooldown == 0:
        if difficulty_level == 0: obstacle_size = 2.0
        elif difficulty_level == 1: obstacle_size = 2.4
        else: obstacle_size = 2.8
        car_radius = 1.2
        obstacle_radius = obstacle_size / 2.0
        collision_distance_squared = (car_radius + obstacle_radius) ** 2
        for ox, _, oz in all_obstacles[track_selection]:
            if (car_x - ox)**2 + (car_z - oz)**2 < collision_distance_squared:
                car_speed *= -0.5; health -= 1; collision_cooldown = 60
                if health <= 0:
                    health = 0
                    if game_start_time > 0:
                        now_t = time.time()
                        current_lap_time = now_t - game_start_time
                        if time_limit != float('inf'):
                            final_time_left = max(0, time_limit - (now_t - game_start_time))
                    game_state = 'game_over'
                    game_over_message = "GAME OVER"
                    return

    if not boost_active:
        for px, _, pz in all_boost_pads[track_selection]:
            if (car_x - px)**2 + (car_z - pz)**2 < 9:
                boost_active = True; boost_timer = time.time()
                car_speed = min(boosted_max_speed, car_speed * boost_multiplier)
    if boost_active and time.time() - boost_timer > boost_duration: boost_active = False

    if START_LINE_P0:
        sx0, sz0 = START_LINE_P0; sx1, sz1 = START_LINE_P1
        side = (sx1 - sx0) * (car_z - sz0) - (sz1 - sz0) * (car_x - sx0)
        crossed_forward = prev_start_side is not None and prev_start_side < 0 and side >= 0 and abs(car_speed) > 0.05
        if crossed_forward:
            now_t = time.time()
            if lap_start_time > 0:
                lap_time = now_t - lap_start_time
            else:
                lap_time = now_t - game_start_time if game_start_time > 0 else 0.0
            current_best = best_times.get(track_selection, float('inf'))
            if lap_time > 0 and lap_time < current_best:
                best_times[track_selection] = lap_time
            lap_count += 1
            lap_start_time = now_t
            if lap_count >= target_laps:
                if game_start_time > 0:
                    now_t = time.time()
                    current_lap_time = now_t - game_start_time
                    if time_limit != float('inf'):
                        final_time_left = max(0, time_limit - (now_t - game_start_time))
                game_state = 'game_over'
                game_over_message = "Congrats! You Won!" if time_limit != float('inf') else "Finished!"
                prev_start_side = side
                return
        prev_start_side = side

    if game_start_time > 0:
        current_lap_time = time.time() - game_start_time
    else:
        current_lap_time = 0.0


    remaining_active = []
    for pickup_pos in nos_pickups[track_selection]:
        if (car_x - pickup_pos[0])**2 + (car_z - pickup_pos[2])**2 < 9 and nos_count < nos_max:
            nos_count += 1
            inactive_nos_pickups.append( (pickup_pos, time.time()) )
        else:
            remaining_active.append(pickup_pos)
    nos_pickups[track_selection] = remaining_active

    respawned = []
    still_inactive = []
    for inactive_pickup in inactive_nos_pickups:
        if time.time() - inactive_pickup[1] > nos_respawn_delay:
            respawned.append(inactive_pickup[0])
        else:
            still_inactive.append(inactive_pickup)
    if respawned:
        nos_pickups[track_selection].extend(respawned)
        inactive_nos_pickups = still_inactive

    if nos_active and time.time() - nos_timer > nos_duration: nos_active = False

def idle():
    """Idle callback to update logic during play."""
    if game_state == 'playing':
        update_game_logic()
    glutPostRedisplay()

def handle_menu_keyboard(key):
    """Handle menu navigation and option changes."""
    global menu_current_option, menu_track_selection, menu_weather_selection, menu_difficulty_selection, menu_lap_selection, menu_time_selection
    num_menu_options = 6
    if key == b'w': menu_current_option = (menu_current_option - 1 + num_menu_options) % num_menu_options
    elif key == b's': menu_current_option = (menu_current_option + 1) % num_menu_options
    elif key in (b'a', b'd'):
        delta = -1 if key == b'a' else 1
        if menu_current_option == 0: menu_track_selection = (menu_track_selection + delta + 4) % 4
        elif menu_current_option == 1: menu_weather_selection = (menu_weather_selection + delta + len(weather_names)) % len(weather_names)
        elif menu_current_option == 2: menu_difficulty_selection = (menu_difficulty_selection + delta + 3) % 3
        elif menu_current_option == 3: menu_lap_selection = (menu_lap_selection + delta + 10) % 10
        elif menu_current_option == 4: menu_time_selection = (menu_time_selection + delta + len(time_limit_options)) % len(time_limit_options)
    elif key == b'\r' and menu_current_option == 5: start_game()

def handle_game_keyboard(key):
    """Handle in-game key presses (movement and NOS)."""
    global nos_active, nos_timer, nos_count, car_speed
    key_str = key.decode("utf-8").lower()
    if key_str in keys: keys[key_str] = True
    if key_str == ' ' and nos_count > 0 and not nos_active:
        nos_active = True; nos_count -= 1; nos_timer = time.time()
        car_speed = min(nos_max_speed, car_speed * nos_multiplier)

def keyboard(key, x, y):
    """Dispatch keyboard input based on game state."""
    if game_state == 'menu':
        handle_menu_keyboard(key)
    elif game_state == 'playing':
        if key == b'\x1b':
            reset_to_menu()
        else:
            handle_game_keyboard(key)
    elif game_state == 'game_over':
        if key == b'r' or key == b'\x1b':
            reset_to_menu()

def keyboard_up(key, x, y):
    """Key-up handler to stop movement flags."""
    key_str = key.decode("utf-8").lower()
    if key_str in keys: keys[key_str] = False

def mouse(button, state, x, y):
    """Toggle camera mode on left click."""
    global camera_mode
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN: camera_mode = 1 - camera_mode

def reset_game_state():
    """Reset gameplay state for a fresh run."""
    global car_speed, lap_start_time, current_lap_time, lap_count, on_lap, prev_start_side, health, collision_cooldown, boost_active, nos_count, nos_active, game_start_time, final_time_left
    car_speed = 0.0
    lap_start_time, current_lap_time, lap_count, on_lap = 0, 0, 0, False
    prev_start_side, health, collision_cooldown, boost_active, nos_count, nos_active = None, 3, 0, False, 0, False
    game_start_time = 0
    final_time_left = None
    init_car_position()

def reset_to_menu():
    """Return to the main menu."""
    global game_state
    game_state = 'menu'

def start_game():
    """Start a new game using current menu selections."""
    global game_state, track_selection, weather_mode, difficulty_level, current_waypoints, inactive_nos_pickups, target_laps, time_limit, game_start_time, final_time_left
    track_selection, weather_mode = menu_track_selection, menu_weather_selection
    difficulty_level = menu_difficulty_selection
    
    target_laps = menu_lap_selection if menu_lap_selection > 0 else float('inf')

    selected_time = time_limit_options[menu_time_selection]
    time_limit = selected_time if selected_time > 0 else float('inf')
    
    game_start_time = 0
    final_time_left = None

    wp_list = [WAYPOINTS_BASE, WAYPOINTS_ALT, WAYPOINTS_T3, WAYPOINTS_T4]
    current_waypoints = wp_list[track_selection]
    
    _build_track_geometry()
    globals()['all_obstacles'], globals()['all_boost_pads'] = _build_obstacles_and_boosts()
    all_nos_options = _build_nos_pickups()
    globals()['nos_pickups'] = [list(track_nos) for track_nos in all_nos_options]
    
    inactive_nos_pickups = []
    
    reset_game_state()
    _setup_precipitation()
    game_state = 'playing'

def main():
    """Set up GLUT and start the main loop."""
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"3D Time Trial Racing Game")
    init()
    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutKeyboardFunc(keyboard)
    glutKeyboardUpFunc(keyboard_up)
    glutMouseFunc(mouse)
    glutMainLoop()

if __name__ == "__main__":
    main()
