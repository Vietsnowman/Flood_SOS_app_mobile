# import sqlite3
# import os
# from datetime import datetime

# DB_PATH = "db/flood_app.db"

# def get_conn():
#     os.makedirs("db", exist_ok=True)
#     return sqlite3.connect(DB_PATH, check_same_thread=False)

# def init_db():
#     conn = get_conn()
#     cur = conn.cursor()

#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS incidents (
#         gid TEXT PRIMARY KEY,
#         status TEXT,
#         last_update TEXT,
#         note TEXT
#     )
#     """)

#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS allocations (
#         gid TEXT PRIMARY KEY,
#         teams INTEGER,
#         boats INTEGER,
#         trucks INTEGER,
#         food_packs INTEGER,
#         water_liters INTEGER,
#         medical_kits INTEGER,
#         last_update TEXT
#     )
#     """)

#     conn.commit()
#     conn.close()

# def get_incident(gid):
#     conn = get_conn()
#     cur = conn.cursor()
#     cur.execute("SELECT gid,status,last_update,note FROM incidents WHERE gid=?", (gid,))
#     row = cur.fetchone()
#     conn.close()
#     return row

# def upsert_incident(gid, status, note=""):
#     conn = get_conn()
#     cur = conn.cursor()
#     now = datetime.now().isoformat(timespec="seconds")
#     cur.execute("""
#         INSERT INTO incidents (gid,status,last_update,note)
#         VALUES (?,?,?,?)
#         ON CONFLICT(gid) DO UPDATE SET
#             status=excluded.status,
#             last_update=excluded.last_update,
#             note=excluded.note
#     """, (gid,status,now,note))
#     conn.commit()
#     conn.close()

# def get_all_incidents():
#     conn = get_conn()
#     cur = conn.cursor()
#     cur.execute("SELECT gid,status,last_update,note FROM incidents")
#     rows = cur.fetchall()
#     conn.close()
#     return rows

# def upsert_allocation(gid, alloc_dict):
#     conn = get_conn()
#     cur = conn.cursor()
#     now = datetime.now().isoformat(timespec="seconds")
#     cur.execute("""
#         INSERT INTO allocations (gid,teams,boats,trucks,food_packs,water_liters,medical_kits,last_update)
#         VALUES (?,?,?,?,?,?,?,?)
#         ON CONFLICT(gid) DO UPDATE SET
#             teams=excluded.teams,
#             boats=excluded.boats,
#             trucks=excluded.trucks,
#             food_packs=excluded.food_packs,
#             water_liters=excluded.water_liters,
#             medical_kits=excluded.medical_kits,
#             last_update=excluded.last_update
#     """, (
#         gid,
#         int(alloc_dict.get("teams",0)),
#         int(alloc_dict.get("boats",0)),
#         int(alloc_dict.get("trucks",0)),
#         int(alloc_dict.get("food_packs",0)),
#         int(alloc_dict.get("water_liters",0)),
#         int(alloc_dict.get("medical_kits",0)),
#         now
#     ))
#     conn.commit()
#     conn.close()

# def get_all_allocations():
#     conn = get_conn()
#     cur = conn.cursor()
#     cur.execute("SELECT gid,teams,boats,trucks,food_packs,water_liters,medical_kits,last_update FROM allocations")
#     rows = cur.fetchall()
#     conn.close()
#     return rows












#===========================================================================================================================================
# # src/db_utils.py
# import os
# import sqlite3
# from datetime import datetime
# from pathlib import Path

# # DB path theo project root (khớp cấu trúc của bạn: project/db/)
# SRC_DIR = Path(__file__).resolve().parent
# PROJECT_DIR = SRC_DIR.parent
# DB_PATH = PROJECT_DIR / "db" / "app.db"
# (DB_PATH.parent).mkdir(parents=True, exist_ok=True)

# def _conn():
#     return sqlite3.connect(DB_PATH, check_same_thread=False)

# def init_db():
#     con = _conn()
#     cur = con.cursor()

#     # ====== (GIỮ NGUYÊN) incidents / allocations nếu bạn đã có ======
#     # Nếu bạn đã tạo rồi thì bỏ qua/đừng trùng tên.
#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS incidents(
#         gid3 TEXT PRIMARY KEY,
#         status TEXT,
#         last_update TEXT,
#         note TEXT
#     )
#     """)

#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS allocations(
#         gid3 TEXT PRIMARY KEY,
#         teams INTEGER,
#         boats INTEGER,
#         trucks INTEGER,
#         food_packs INTEGER,
#         water_liters INTEGER,
#         medical_kits INTEGER,
#         last_update TEXT
#     )
#     """)

#     # ====== (MỚI) requests: tiếp nhận yêu cầu ======
#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS requests(
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         created_at TEXT,
#         caller_name TEXT,
#         phone TEXT,
#         gid3 TEXT,
#         lat REAL,
#         lon REAL,
#         people INTEGER,
#         urgency INTEGER,
#         note TEXT,
#         status TEXT,          -- MỚI / ĐÃ TẠO NHIỆM VỤ / ĐANG XỬ LÝ / ĐÃ XONG / HỦY
#         linked_task_id INTEGER
#     )
#     """)

#     # ====== (MỚI) tasks: nhiệm vụ điều phối ======
#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS tasks(
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         created_at TEXT,
#         gid3 TEXT,
#         commune_name TEXT,
#         task_type TEXT,       -- SƠ TÁN / CỨU HỘ / CẤP PHÁT / KHẢO SÁT / Y TẾ
#         priority_score REAL,
#         assigned_team TEXT,   -- tên đội/kíp
#         boats INTEGER,
#         trucks INTEGER,
#         status TEXT,          -- MỚI / ĐÃ GIAO / ĐANG ĐI / ĐANG THỰC HIỆN / HOÀN TẤT / HỦY
#         eta_min INTEGER,
#         start_time TEXT,
#         end_time TEXT,
#         note TEXT,
#         source_request_id INTEGER
#     )
#     """)

#     # ====== (MỚI) shelter_status: tình trạng điểm trú ẩn ======
#     # shelter_id dùng theo shelters.csv: nếu có cột id thì dùng; nếu không có thì dùng name (mã hoá)
#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS shelter_status(
#         shelter_key TEXT PRIMARY KEY,  -- key duy nhất
#         name TEXT,
#         gid3 TEXT,
#         capacity INTEGER,
#         current_people INTEGER,
#         need_food INTEGER,
#         need_water INTEGER,
#         need_med INTEGER,
#         note TEXT,
#         last_update TEXT
#     )
#     """)

#     # ====== (MỚI) shelter_moves: lịch sử chuyển người ======
#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS shelter_moves(
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         created_at TEXT,
#         from_gid3 TEXT,
#         to_shelter_key TEXT,
#         people INTEGER,
#         task_id INTEGER,
#         note TEXT
#     )
#     """)

#     con.commit()
#     con.close()


# # =========================
# # REQUESTS
# # =========================
# def create_request(payload: dict) -> int:
#     con = _conn()
#     cur = con.cursor()
#     now = datetime.now().isoformat(timespec="seconds")
#     cur.execute("""
#         INSERT INTO requests(created_at, caller_name, phone, gid3, lat, lon, people, urgency, note, status, linked_task_id)
#         VALUES(?,?,?,?,?,?,?,?,?,?,?)
#     """, (
#         now,
#         payload.get("caller_name"),
#         payload.get("phone"),
#         payload.get("gid3"),
#         payload.get("lat"),
#         payload.get("lon"),
#         int(payload.get("people") or 0),
#         int(payload.get("urgency") or 3),
#         payload.get("note"),
#         payload.get("status") or "MỚI",
#         None
#     ))
#     rid = cur.lastrowid
#     con.commit()
#     con.close()
#     return rid

# def list_requests(limit=200, status=None):
#     con = _conn()
#     cur = con.cursor()
#     if status:
#         cur.execute("""
#             SELECT id, created_at, caller_name, phone, gid3, lat, lon, people, urgency, note, status, linked_task_id
#             FROM requests WHERE status=? ORDER BY id DESC LIMIT ?
#         """, (status, limit))
#     else:
#         cur.execute("""
#             SELECT id, created_at, caller_name, phone, gid3, lat, lon, people, urgency, note, status, linked_task_id
#             FROM requests ORDER BY id DESC LIMIT ?
#         """, (limit,))
#     rows = cur.fetchall()
#     con.close()
#     return rows

# def update_request_status(request_id: int, status: str, linked_task_id=None):
#     con = _conn()
#     cur = con.cursor()
#     cur.execute("""
#         UPDATE requests SET status=?, linked_task_id=COALESCE(?, linked_task_id)
#         WHERE id=?
#     """, (status, linked_task_id, request_id))
#     con.commit()
#     con.close()


# # =========================
# # TASKS
# # =========================
# def create_task(payload: dict) -> int:
#     con = _conn()
#     cur = con.cursor()
#     now = datetime.now().isoformat(timespec="seconds")
#     cur.execute("""
#         INSERT INTO tasks(created_at, gid3, commune_name, task_type, priority_score,
#                           assigned_team, boats, trucks, status, eta_min,
#                           start_time, end_time, note, source_request_id)
#         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
#     """, (
#         now,
#         payload.get("gid3"),
#         payload.get("commune_name"),
#         payload.get("task_type"),
#         float(payload.get("priority_score") or 0),
#         payload.get("assigned_team"),
#         int(payload.get("boats") or 0),
#         int(payload.get("trucks") or 0),
#         payload.get("status") or "MỚI",
#         int(payload.get("eta_min") or 0),
#         payload.get("start_time"),
#         payload.get("end_time"),
#         payload.get("note"),
#         payload.get("source_request_id")
#     ))
#     tid = cur.lastrowid
#     con.commit()
#     con.close()
#     return tid

# def list_tasks(limit=300, status=None):
#     con = _conn()
#     cur = con.cursor()
#     if status:
#         cur.execute("""
#             SELECT id, created_at, gid3, commune_name, task_type, priority_score,
#                    assigned_team, boats, trucks, status, eta_min, start_time, end_time, note, source_request_id
#             FROM tasks WHERE status=? ORDER BY id DESC LIMIT ?
#         """, (status, limit))
#     else:
#         cur.execute("""
#             SELECT id, created_at, gid3, commune_name, task_type, priority_score,
#                    assigned_team, boats, trucks, status, eta_min, start_time, end_time, note, source_request_id
#             FROM tasks ORDER BY id DESC LIMIT ?
#         """, (limit,))
#     rows = cur.fetchall()
#     con.close()
#     return rows

# def update_task(task_id: int, status: str, assigned_team=None, boats=None, trucks=None, eta_min=None, note=None, start_time=None, end_time=None):
#     con = _conn()
#     cur = con.cursor()
#     cur.execute("""
#         UPDATE tasks
#         SET status=?,
#             assigned_team=COALESCE(?, assigned_team),
#             boats=COALESCE(?, boats),
#             trucks=COALESCE(?, trucks),
#             eta_min=COALESCE(?, eta_min),
#             note=COALESCE(?, note),
#             start_time=COALESCE(?, start_time),
#             end_time=COALESCE(?, end_time)
#         WHERE id=?
#     """, (status, assigned_team, boats, trucks, eta_min, note, start_time, end_time, task_id))
#     con.commit()
#     con.close()


# # =========================
# # SHELTER STATUS + MOVES
# # =========================
# def upsert_shelter_status(payload: dict):
#     con = _conn()
#     cur = con.cursor()
#     now = datetime.now().isoformat(timespec="seconds")
#     cur.execute("""
#         INSERT INTO shelter_status(shelter_key, name, gid3, capacity, current_people,
#                                    need_food, need_water, need_med, note, last_update)
#         VALUES(?,?,?,?,?,?,?,?,?,?)
#         ON CONFLICT(shelter_key) DO UPDATE SET
#             name=excluded.name,
#             gid3=excluded.gid3,
#             capacity=excluded.capacity,
#             current_people=excluded.current_people,
#             need_food=excluded.need_food,
#             need_water=excluded.need_water,
#             need_med=excluded.need_med,
#             note=excluded.note,
#             last_update=excluded.last_update
#     """, (
#         payload["shelter_key"],
#         payload.get("name"),
#         payload.get("gid3"),
#         int(payload.get("capacity") or 0),
#         int(payload.get("current_people") or 0),
#         int(payload.get("need_food") or 0),
#         int(payload.get("need_water") or 0),
#         int(payload.get("need_med") or 0),
#         payload.get("note"),
#         now
#     ))
#     con.commit()
#     con.close()

# def list_shelter_status(limit=500):
#     con = _conn()
#     cur = con.cursor()
#     cur.execute("""
#         SELECT shelter_key, name, gid3, capacity, current_people, need_food, need_water, need_med, note, last_update
#         FROM shelter_status ORDER BY last_update DESC LIMIT ?
#     """, (limit,))
#     rows = cur.fetchall()
#     con.close()
#     return rows

# def create_shelter_move(payload: dict):
#     con = _conn()
#     cur = con.cursor()
#     now = datetime.now().isoformat(timespec="seconds")
#     cur.execute("""
#         INSERT INTO shelter_moves(created_at, from_gid3, to_shelter_key, people, task_id, note)
#         VALUES(?,?,?,?,?,?)
#     """, (
#         now,
#         payload.get("from_gid3"),
#         payload.get("to_shelter_key"),
#         int(payload.get("people") or 0),
#         payload.get("task_id"),
#         payload.get("note")
#     ))
#     con.commit()
#     con.close()

# def list_shelter_moves(limit=300):
#     con = _conn()
#     cur = con.cursor()
#     cur.execute("""
#         SELECT id, created_at, from_gid3, to_shelter_key, people, task_id, note
#         FROM shelter_moves ORDER BY id DESC LIMIT ?
#     """, (limit,))
#     rows = cur.fetchall()
#     con.close()
#     return rows




# src/db_utils.py
import sqlite3
from datetime import datetime
from pathlib import Path

# =========================
# DB PATH (THỐNG NHẤT)
# =========================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DB_PATH = PROJECT_DIR / "db" / "app.db"
(DB_PATH.parent).mkdir(parents=True, exist_ok=True)


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# =========================
# INIT DB
# =========================
def init_db():
    con = _conn()
    cur = con.cursor()

    # -------- INCIDENTS --------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS incidents(
        gid3 TEXT PRIMARY KEY,
        status TEXT,
        last_update TEXT,
        note TEXT
    )
    """)

    # -------- ALLOCATIONS --------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS allocations(
        gid3 TEXT PRIMARY KEY,
        teams INTEGER,
        boats INTEGER,
        trucks INTEGER,
        food_packs INTEGER,
        water_liters INTEGER,
        medical_kits INTEGER,
        last_update TEXT
    )
    """)

    # -------- REQUESTS (HOTLINE) --------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS requests(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        caller_name TEXT,
        phone TEXT,
        gid3 TEXT,
        lat REAL,
        lon REAL,
        people INTEGER,
        urgency INTEGER,
        note TEXT,
        status TEXT,
        linked_task_id INTEGER
    )
    """)

    # -------- TASKS --------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        gid3 TEXT,
        commune_name TEXT,
        task_type TEXT,
        priority_score REAL,
        assigned_team TEXT,
        boats INTEGER,
        trucks INTEGER,
        status TEXT,
        eta_min INTEGER,
        start_time TEXT,
        end_time TEXT,
        note TEXT,
        source_request_id INTEGER
    )
    """)

    # -------- SHELTER STATUS --------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS shelter_status(
        shelter_key TEXT PRIMARY KEY,
        name TEXT,
        gid3 TEXT,
        capacity INTEGER,
        current_people INTEGER,
        need_food INTEGER,
        need_water INTEGER,
        need_med INTEGER,
        note TEXT,
        last_update TEXT
    )
    """)

    # -------- SHELTER MOVES --------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS shelter_moves(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        from_gid3 TEXT,
        to_shelter_key TEXT,
        people INTEGER,
        task_id INTEGER,
        note TEXT
    )
    """)

    con.commit()
    con.close()


# =========================
# INCIDENTS
# =========================
def get_incident(gid3):
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "SELECT gid3, status, last_update, note FROM incidents WHERE gid3=?",
        (gid3,)
    )
    row = cur.fetchone()
    con.close()
    return row


def upsert_incident(gid3, status, note=""):
    con = _conn()
    cur = con.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO incidents (gid3, status, last_update, note)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(gid3) DO UPDATE SET
            status=excluded.status,
            last_update=excluded.last_update,
            note=excluded.note
    """, (gid3, status, now, note))
    con.commit()
    con.close()


def get_all_incidents():
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT gid3, status, last_update, note FROM incidents")
    rows = cur.fetchall()
    con.close()
    return rows


# =========================
# ALLOCATIONS
# =========================
def upsert_allocation(gid3, alloc):
    con = _conn()
    cur = con.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO allocations(gid3, teams, boats, trucks, food_packs, water_liters, medical_kits, last_update)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(gid3) DO UPDATE SET
            teams=excluded.teams,
            boats=excluded.boats,
            trucks=excluded.trucks,
            food_packs=excluded.food_packs,
            water_liters=excluded.water_liters,
            medical_kits=excluded.medical_kits,
            last_update=excluded.last_update
    """, (
        gid3,
        int(alloc.get("teams", 0)),
        int(alloc.get("boats", 0)),
        int(alloc.get("trucks", 0)),
        int(alloc.get("food_packs", 0)),
        int(alloc.get("water_liters", 0)),
        int(alloc.get("medical_kits", 0)),
        now
    ))
    con.commit()
    con.close()


def get_all_allocations():
    con = _conn()
    cur = con.cursor()
    cur.execute("""
        SELECT gid3, teams, boats, trucks,
               food_packs, water_liters, medical_kits, last_update
        FROM allocations
    """)
    rows = cur.fetchall()
    con.close()
    return rows


# =========================
# REQUESTS (HOTLINE)
# =========================
def create_request(payload):
    con = _conn()
    cur = con.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO requests(created_at, caller_name, phone, gid3, lat, lon,
                             people, urgency, note, status, linked_task_id)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
    """, (
        now,
        payload.get("caller_name"),
        payload.get("phone"),
        payload.get("gid3"),
        payload.get("lat"),
        payload.get("lon"),
        int(payload.get("people") or 0),
        int(payload.get("urgency") or 3),
        payload.get("note"),
        payload.get("status", "MỚI"),
        None
    ))
    rid = cur.lastrowid
    con.commit()
    con.close()
    return rid


def list_requests(limit=200):
    con = _conn()
    cur = con.cursor()
    cur.execute("""
        SELECT id, created_at, caller_name, phone, gid3, lat, lon,
               people, urgency, note, status, linked_task_id
        FROM requests
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    con.close()
    return rows


def update_request_status(request_id, status, linked_task_id=None):
    con = _conn()
    cur = con.cursor()
    cur.execute("""
        UPDATE requests
        SET status=?, linked_task_id=COALESCE(?, linked_task_id)
        WHERE id=?
    """, (status, linked_task_id, request_id))
    con.commit()
    con.close()


# =========================
# TASKS
# =========================
def create_task(payload):
    con = _conn()
    cur = con.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO tasks(created_at, gid3, commune_name, task_type, priority_score,
                          assigned_team, boats, trucks, status, eta_min,
                          start_time, end_time, note, source_request_id)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        now,
        payload.get("gid3"),
        payload.get("commune_name"),
        payload.get("task_type"),
        float(payload.get("priority_score") or 0),
        payload.get("assigned_team"),
        int(payload.get("boats") or 0),
        int(payload.get("trucks") or 0),
        payload.get("status", "MỚI"),
        int(payload.get("eta_min") or 0),
        payload.get("start_time"),
        payload.get("end_time"),
        payload.get("note"),
        payload.get("source_request_id")
    ))
    tid = cur.lastrowid
    con.commit()
    con.close()
    return tid


def list_tasks(limit=300):
    con = _conn()
    cur = con.cursor()
    cur.execute("""
        SELECT id, created_at, gid3, commune_name, task_type, priority_score,
               assigned_team, boats, trucks, status, eta_min,
               start_time, end_time, note, source_request_id
        FROM tasks ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    con.close()
    return rows


def update_task(task_id, status, assigned_team=None, boats=None, trucks=None,
                eta_min=None, note=None, start_time=None, end_time=None):
    con = _conn()
    cur = con.cursor()
    cur.execute("""
        UPDATE tasks
        SET status=?,
            assigned_team=COALESCE(?, assigned_team),
            boats=COALESCE(?, boats),
            trucks=COALESCE(?, trucks),
            eta_min=COALESCE(?, eta_min),
            note=COALESCE(?, note),
            start_time=COALESCE(?, start_time),
            end_time=COALESCE(?, end_time)
        WHERE id=?
    """, (status, assigned_team, boats, trucks, eta_min, note, start_time, end_time, task_id))
    con.commit()
    con.close()


# =========================
# SHELTER STATUS + MOVES
# =========================
def upsert_shelter_status(payload):
    con = _conn()
    cur = con.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO shelter_status(shelter_key, name, gid3, capacity, current_people,
                                   need_food, need_water, need_med, note, last_update)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(shelter_key) DO UPDATE SET
            name=excluded.name,
            gid3=excluded.gid3,
            capacity=excluded.capacity,
            current_people=excluded.current_people,
            need_food=excluded.need_food,
            need_water=excluded.need_water,
            need_med=excluded.need_med,
            note=excluded.note,
            last_update=excluded.last_update
    """, (
        payload["shelter_key"],
        payload.get("name"),
        payload.get("gid3"),
        int(payload.get("capacity") or 0),
        int(payload.get("current_people") or 0),
        int(payload.get("need_food") or 0),
        int(payload.get("need_water") or 0),
        int(payload.get("need_med") or 0),
        payload.get("note"),
        now
    ))
    con.commit()
    con.close()


def list_shelter_status(limit=500):
    con = _conn()
    cur = con.cursor()
    cur.execute("""
        SELECT shelter_key, name, gid3, capacity, current_people,
               need_food, need_water, need_med, note, last_update
        FROM shelter_status
        ORDER BY last_update DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    con.close()
    return rows


def create_shelter_move(payload):
    con = _conn()
    cur = con.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO shelter_moves(created_at, from_gid3, to_shelter_key, people, task_id, note)
        VALUES(?,?,?,?,?,?)
    """, (
        now,
        payload.get("from_gid3"),
        payload.get("to_shelter_key"),
        int(payload.get("people") or 0),
        payload.get("task_id"),
        payload.get("note")
    ))
    con.commit()
    con.close()


def list_shelter_moves(limit=300):
    con = _conn()
    cur = con.cursor()
    cur.execute("""
        SELECT id, created_at, from_gid3, to_shelter_key, people, task_id, note
        FROM shelter_moves ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    con.close()
    return rows
