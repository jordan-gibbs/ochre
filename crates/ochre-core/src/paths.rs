use std::path::{Path, PathBuf};

use directories::ProjectDirs;

/// The app's directory name under the OS config / data roots.
pub const APP_DIR: &str = "ochre";
/// The name used before the rename to Ochre; [`migrate_legacy_dirs`] moves it once.
pub const LEGACY_APP_DIR: &str = "openwhisprflow";

fn dirs() -> ProjectDirs {
    ProjectDirs::from("", "", APP_DIR).expect("no home directory")
}

/// `config.toml` lives here.
pub fn config_dir() -> PathBuf {
    std::env::var_os("OCHRE_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs().config_dir().to_path_buf())
}

/// History, runtime file, logs.
pub fn data_dir() -> PathBuf {
    std::env::var_os("OCHRE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs().data_local_dir().to_path_buf())
}

/// Downloaded model weights and helper binaries (llama-server).
pub fn models_dir() -> PathBuf {
    std::env::var_os("OCHRE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir().join("models"))
}

pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

/// The per-app roots that hold everything: on Windows `ProjectDirs` adds a `config` / `data`
/// subfolder, so the root is its parent (which also catches siblings like `logs`); elsewhere the
/// dirs are the roots. Deduplicated (macOS keeps config and data in the same place).
fn roots(p: &ProjectDirs) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for d in [p.config_dir(), p.data_local_dir()] {
        let root = if cfg!(windows) {
            d.parent().unwrap_or(d).to_path_buf()
        } else {
            d.to_path_buf()
        };
        if !out.contains(&root) {
            out.push(root);
        }
    }
    out
}

/// What [`migrate_legacy_dirs`] did with one legacy directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Migration {
    Renamed {
        from: PathBuf,
        to: PathBuf,
    },
    Copied {
        from: PathBuf,
        to: PathBuf,
    },
    Failed {
        from: PathBuf,
        to: PathBuf,
        error: String,
    },
}

/// One-time move from the pre-rename `openwhisprflow` dirs to `ochre`: for each root, if the new
/// one doesn't exist and the old one does, rename it (instant, same volume), falling back to a
/// recursive copy. Skipped when any directory override env var is set (tests, portable setups).
/// Call once at startup, before anything opens the config, history or models.
pub fn migrate_legacy_dirs() -> Vec<Migration> {
    if ["OCHRE_CONFIG_DIR", "OCHRE_DATA_DIR", "OCHRE_MODELS_DIR"]
        .iter()
        .any(|v| std::env::var_os(v).is_some())
    {
        return Vec::new();
    }
    let (Some(old), Some(new)) = (
        ProjectDirs::from("", "", LEGACY_APP_DIR),
        ProjectDirs::from("", "", APP_DIR),
    ) else {
        return Vec::new();
    };
    let done: Vec<Migration> = roots(&old)
        .into_iter()
        .zip(roots(&new))
        .filter_map(|(from, to)| migrate_dir(&from, &to))
        .collect();
    for m in &done {
        match m {
            Migration::Renamed { from, to } => {
                tracing::info!(from = %from.display(), to = %to.display(), "migrated app data dir (renamed)")
            }
            Migration::Copied { from, to } => {
                tracing::info!(from = %from.display(), to = %to.display(), "migrated app data dir (copied; old dir left in place)")
            }
            Migration::Failed { from, to, error } => {
                tracing::warn!(from = %from.display(), to = %to.display(), %error, "could not migrate app data dir")
            }
        }
    }
    done
}

/// Move `from` to `to` if `to` is missing and `from` exists. `None` when there is nothing to do.
pub fn migrate_dir(from: &Path, to: &Path) -> Option<Migration> {
    if to.exists() || !from.is_dir() {
        return None;
    }
    let (from_b, to_b) = (from.to_path_buf(), to.to_path_buf());
    if let Some(parent) = to.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::rename(from, to).is_ok() {
        return Some(Migration::Renamed {
            from: from_b,
            to: to_b,
        });
    }
    match copy_tree(from, to) {
        Ok(()) => Some(Migration::Copied {
            from: from_b,
            to: to_b,
        }),
        Err(e) => {
            let _ = std::fs::remove_dir_all(to); // don't leave a half copy that blocks a retry
            Some(Migration::Failed {
                from: from_b,
                to: to_b,
                error: e.to_string(),
            })
        }
    }
}

fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let dest = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &dest)?;
        } else {
            std::fs::copy(entry.path(), &dest)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_dir_moves_once_and_never_overwrites() {
        let base = std::env::temp_dir().join(format!("ochre-migrate-{}", uuid::Uuid::new_v4()));
        let old = base.join("openwhisprflow");
        let new = base.join("ochre");
        std::fs::create_dir_all(old.join("config")).unwrap();
        std::fs::write(old.join("config").join("config.toml"), "x = 1\n").unwrap();

        let m = migrate_dir(&old, &new).expect("migrated");
        assert!(matches!(m, Migration::Renamed { .. }), "{m:?}");
        assert!(!old.exists());
        assert_eq!(
            std::fs::read_to_string(new.join("config").join("config.toml")).unwrap(),
            "x = 1\n"
        );

        // new already exists: the old dir is left alone
        std::fs::create_dir_all(&old).unwrap();
        assert_eq!(migrate_dir(&old, &new), None);
        assert!(old.exists());
        // nothing to migrate
        assert_eq!(
            migrate_dir(&base.join("missing"), &base.join("other")),
            None
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn copy_tree_copies_nested_files() {
        let base = std::env::temp_dir().join(format!("ochre-copy-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(base.join("a").join("b")).unwrap();
        std::fs::write(base.join("a").join("b").join("f.txt"), "hi").unwrap();
        copy_tree(&base.join("a"), &base.join("c")).unwrap();
        assert_eq!(
            std::fs::read_to_string(base.join("c").join("b").join("f.txt")).unwrap(),
            "hi"
        );
        let _ = std::fs::remove_dir_all(&base);
    }
}
