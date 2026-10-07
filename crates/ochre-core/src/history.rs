//! Local dictation history: text only, never audio. Written *before* insertion, so nothing is
//! lost if focus changed mid-dictation (the save-before-insert rule).

use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
CREATE TABLE IF NOT EXISTS dictations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created REAL NOT NULL,
  raw TEXT NOT NULL,
  text TEXT NOT NULL,
  app TEXT NOT NULL DEFAULT '',
  stt TEXT NOT NULL DEFAULT '',
  refiner TEXT NOT NULL DEFAULT '',
  duration_ms INTEGER NOT NULL DEFAULT 0,
  inserted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS dictations_created ON dictations(created);
";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub id: i64,
    /// Unix seconds.
    pub created: f64,
    pub raw: String,
    pub text: String,
    pub app: String,
    pub stt: String,
    pub refiner: String,
    pub duration_ms: u64,
    pub inserted: bool,
    /// Why the text is what it is, when that is not the obvious refined/raw: "redictation" =
    /// typed as heard because the same thing was just dictated again (the cleanup was wrong).
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, Default)]
pub struct NewEntry<'a> {
    pub raw: &'a str,
    pub text: &'a str,
    pub app: &'a str,
    pub stt: &'a str,
    pub refiner: &'a str,
    pub duration_ms: u64,
    pub note: &'a str,
}

/// `None` connection = history disabled; every call is a cheap no-op.
pub struct History {
    db: Option<Mutex<Connection>>,
}

fn db_err(e: rusqlite::Error) -> Error {
    Error::Other(format!("history: {e}"))
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

impl History {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path).map_err(db_err)?;
        conn.execute_batch(SCHEMA).map_err(db_err)?;
        // Added after the first release: older files get the column (empty for existing rows).
        let has_note = conn
            .prepare("SELECT 1 FROM pragma_table_info('dictations') WHERE name = 'note'")
            .and_then(|mut s| s.exists([]))
            .map_err(db_err)?;
        if !has_note {
            conn.execute_batch("ALTER TABLE dictations ADD COLUMN note TEXT NOT NULL DEFAULT ''")
                .map_err(db_err)?;
        }
        Ok(Self {
            db: Some(Mutex::new(conn)),
        })
    }

    pub fn disabled() -> Self {
        Self { db: None }
    }

    pub fn add(&self, e: NewEntry<'_>) -> Result<Option<i64>> {
        let Some(db) = &self.db else { return Ok(None) };
        let db = db.lock().unwrap();
        db.execute(
            "INSERT INTO dictations(created, raw, text, app, stt, refiner, duration_ms, note) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![now(), e.raw, e.text, e.app, e.stt, e.refiner, e.duration_ms as i64, e.note],
        )
        .map_err(db_err)?;
        Ok(Some(db.last_insert_rowid()))
    }

    pub fn mark_inserted(&self, id: i64) -> Result<()> {
        if let Some(db) = &self.db {
            db.lock()
                .unwrap()
                .execute("UPDATE dictations SET inserted = 1 WHERE id = ?1", [id])
                .map_err(db_err)?;
        }
        Ok(())
    }

    pub fn query(&self, q: &str, limit: usize) -> Result<Vec<Entry>> {
        let Some(db) = &self.db else {
            return Ok(Vec::new());
        };
        let db = db.lock().unwrap();
        let like = format!("%{}%", q.replace('%', "\\%").replace('_', "\\_"));
        let mut stmt = db
            .prepare(
                "SELECT id, created, raw, text, app, stt, refiner, duration_ms, inserted, note FROM dictations
                 WHERE ?1 = '' OR text LIKE ?2 ESCAPE '\\' OR raw LIKE ?2 ESCAPE '\\'
                 ORDER BY id DESC LIMIT ?3",
            )
            .map_err(db_err)?;
        let rows = stmt
            .query_map(params![q, like, limit.clamp(1, 500) as i64], |r| {
                Ok(Entry {
                    id: r.get(0)?,
                    created: r.get(1)?,
                    raw: r.get(2)?,
                    text: r.get(3)?,
                    app: r.get(4)?,
                    stt: r.get(5)?,
                    refiner: r.get(6)?,
                    duration_ms: r.get::<_, i64>(7)? as u64,
                    inserted: r.get::<_, i64>(8)? != 0,
                    note: r.get(9)?,
                })
            })
            .map_err(db_err)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(db_err)
    }

    pub fn prune(&self, keep_days: u32) -> Result<()> {
        if let (Some(db), true) = (&self.db, keep_days > 0) {
            let cutoff = now() - f64::from(keep_days) * 86_400.0;
            db.lock()
                .unwrap()
                .execute("DELETE FROM dictations WHERE created < ?1", [cutoff])
                .map_err(db_err)?;
        }
        Ok(())
    }

    pub fn clear(&self) -> Result<()> {
        if let Some(db) = &self.db {
            db.lock()
                .unwrap()
                .execute("DELETE FROM dictations", [])
                .map_err(db_err)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_query_mark_clear() {
        let path =
            std::env::temp_dir().join(format!("ochre-hist-{}.sqlite3", uuid::Uuid::new_v4()));
        let h = History::open(&path).unwrap();
        let a = h
            .add(NewEntry {
                raw: "um hello",
                text: "Hello.",
                app: "slack",
                ..Default::default()
            })
            .unwrap()
            .unwrap();
        h.add(NewEntry {
            raw: "second",
            text: "Second.",
            note: "redictation",
            ..Default::default()
        })
        .unwrap();
        h.mark_inserted(a).unwrap();
        let all = h.query("", 50).unwrap();
        assert_eq!(
            all.iter().map(|e| e.text.as_str()).collect::<Vec<_>>(),
            ["Second.", "Hello."]
        );
        assert!(all[1].inserted && !all[0].inserted);
        assert_eq!(
            (all[0].note.as_str(), all[1].note.as_str()),
            ("redictation", "")
        );
        assert_eq!(h.query("hello", 50).unwrap().len(), 1);
        assert_eq!(h.query("100%", 50).unwrap().len(), 0);
        h.clear().unwrap();
        assert!(h.query("", 50).unwrap().is_empty());
        drop(h);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn old_files_get_the_note_column() {
        let path =
            std::env::temp_dir().join(format!("ochre-hist-{}.sqlite3", uuid::Uuid::new_v4()));
        {
            let c = Connection::open(&path).unwrap();
            c.execute_batch(
                "CREATE TABLE dictations (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
                 raw TEXT NOT NULL, text TEXT NOT NULL, app TEXT NOT NULL DEFAULT '',
                 stt TEXT NOT NULL DEFAULT '', refiner TEXT NOT NULL DEFAULT '',
                 duration_ms INTEGER NOT NULL DEFAULT 0, inserted INTEGER NOT NULL DEFAULT 0);
                 INSERT INTO dictations(created, raw, text) VALUES (1.0, 'old', 'Old.');",
            )
            .unwrap();
        }
        let h = History::open(&path).unwrap();
        assert_eq!(h.query("", 5).unwrap()[0].note, "");
        drop(h);
        let h = History::open(&path).unwrap(); // idempotent
        assert_eq!(h.query("", 5).unwrap()[0].text, "Old.");
        drop(h);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn disabled_is_noop() {
        let h = History::disabled();
        assert_eq!(h.add(NewEntry::default()).unwrap(), None);
        assert!(h.query("", 10).unwrap().is_empty());
    }
}
