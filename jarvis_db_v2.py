# db.py
import os, sqlite3, uuid, time
from pathlib import Path

def _db_path():
    # read at call time: the server imports this module before load_dotenv()
    return Path(os.getenv("JARVIS_DB_PATH") or "jarvis.db")

def conn():
    c = sqlite3.connect(_db_path())
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with conn() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            username TEXT,
            title TEXT,
            is_pinned INTEGER DEFAULT 0,
            created_at INTEGER,
            updated_at INTEGER
        )""")
        
        # Migration 1: Add is_pinned (will fail safely if it already exists)
        try:
            db.execute("ALTER TABLE chats ADD COLUMN is_pinned INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
            
        # Migration 2: Add username (will fail safely if it already exists)
        try:
            db.execute("ALTER TABLE chats ADD COLUMN username TEXT")
        except sqlite3.OperationalError:
            pass
            
        # Migration 3: Assign all old/unclaimed chats to the default user
        try:
            db.execute("UPDATE chats SET username=? WHERE username IS NULL",
                       (os.getenv("JARVIS_DEFAULT_USER", "owner"),))
        except sqlite3.OperationalError as e:
            print(f"DB Update Error: {e}")

        db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            html TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(chat_id) REFERENCES chats(id)
        )""")

def create_chat(username, title="New chat"):
    chat_id = str(uuid.uuid4())
    now = int(time.time())
    with conn() as db:
        db.execute(
            "INSERT INTO chats(id,username,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (chat_id, username, title, now, now)
        )
    return chat_id

def list_chats(username):
    with conn() as db:
        rows = db.execute(
            "SELECT * FROM chats WHERE username=? ORDER BY is_pinned DESC, updated_at DESC",
            (username,)
        ).fetchall()
    return [dict(r) for r in rows]

def get_chat_name(chat_id: str):
    """Retrieves a single chat's details by ID."""
    with conn() as db:
        chat = db.execute("SELECT id, title FROM chats WHERE id=?", (chat_id,)).fetchone()
    return dict(chat) if chat else None

def get_messages(chat_id, limit=200):
    with conn() as db:
        rows = db.execute("""
            SELECT role, content, html, created_at
            FROM messages
            WHERE chat_id=?
            ORDER BY id ASC
            LIMIT ?
        """, (chat_id, limit)).fetchall()
    return [dict(r) for r in rows]

def add_message(chat_id, role, content, html=None):
    now = int(time.time())
    with conn() as db:
        db.execute(
            "INSERT INTO messages(chat_id,role,content,html,created_at) VALUES(?,?,?,?,?)",
            (chat_id, role, content, html, now)
        )
        db.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))

def rename_chat(chat_id: str, title: str):
    now = int(time.time())
    with conn() as db:
        db.execute("UPDATE chats SET title=?, updated_at=? WHERE id=?", (title, now, chat_id))

def delete_chat(chat_id: str):
    with conn() as db:
        db.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        db.execute("DELETE FROM chats WHERE id=?", (chat_id,))

def maybe_autotitle(chat_id):
    """Optional: use first user message as title if title is still 'New chat'."""
    with conn() as db:
        chat = db.execute("SELECT title FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not chat:
            return
        if (chat["title"] or "").strip().lower() != "new chat":
            return

        first_user = db.execute("""
            SELECT content FROM messages
            WHERE chat_id=? AND role='user'
            ORDER BY id ASC LIMIT 1
        """, (chat_id,)).fetchone()

        if first_user and first_user["content"]:
            title = first_user["content"].strip().splitlines()[0][:42]
            db.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))

def toggle_pin_chat(chat_id: str):
    with conn() as db:
        current = db.execute("SELECT is_pinned FROM chats WHERE id=?", (chat_id,)).fetchone()
        if current:
            # Flip 0 to 1, or 1 to 0
            new_status = 1 if current["is_pinned"] == 0 else 0
            db.execute("UPDATE chats SET is_pinned=? WHERE id=?", (new_status, chat_id))
            return new_status
    return None