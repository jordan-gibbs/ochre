//! API keys: OS keyring first (Windows Credential Manager, macOS Keychain, Secret Service),
//! environment variables as fallback (SPEC §8). Keys are never logged or written to config.

const SERVICE: &str = "ochre";
/// Keyring service used before the rename to Ochre; read as a fallback so stored keys keep working.
const LEGACY_SERVICE: &str = "openwhisprflow";

/// provider -> env vars checked (in order) after `OCHRE_<PROVIDER>_API_KEY`.
pub const PROVIDERS: &[(&str, &[&str])] = &[
    ("openai", &["OPENAI_API_KEY"]),
    ("groq", &["GROQ_API_KEY"]),
    ("soniox", &["SONIOX_API_KEY"]),
    ("deepgram", &["DEEPGRAM_API_KEY"]),
    ("elevenlabs", &["ELEVENLABS_API_KEY", "XI_API_KEY"]),
    ("assemblyai", &["ASSEMBLYAI_API_KEY"]),
    ("anthropic", &["ANTHROPIC_API_KEY"]),
    ("gemini", &["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    ("openrouter", &["OPENROUTER_API_KEY"]),
];

/// Engine ids that share another provider's key: Google's speech engine (`google`, Gemini 3.5
/// Transcribe) uses the same Gemini API key as the `gemini` refiner, so one pasted key configures
/// both stages of the Google connector.
pub const ALIASES: &[(&str, &str)] = &[("google", "gemini")];

/// The provider a key is actually stored under.
pub fn canonical(provider: &str) -> &str {
    ALIASES
        .iter()
        .find(|(a, _)| *a == provider)
        .map(|(_, p)| *p)
        .unwrap_or(provider)
}

fn entry(provider: &str) -> Option<keyring::Entry> {
    keyring::Entry::new(SERVICE, canonical(provider)).ok()
}

fn legacy_entry(provider: &str) -> Option<keyring::Entry> {
    keyring::Entry::new(LEGACY_SERVICE, canonical(provider)).ok()
}

pub fn get(provider: &str) -> Option<String> {
    let provider = canonical(provider);
    if let Some(key) = [entry(provider), legacy_entry(provider)]
        .into_iter()
        .flatten()
        .find_map(|e| e.get_password().ok().filter(|k| !k.is_empty()))
    {
        return Some(key);
    }
    let own = env_var(provider);
    let fallbacks = PROVIDERS
        .iter()
        .find(|(p, _)| *p == provider)
        .map(|(_, v)| *v)
        .unwrap_or(&[]);
    std::iter::once(own.as_str())
        .chain(fallbacks.iter().copied())
        .find_map(|var| std::env::var(var).ok().filter(|v| !v.is_empty()))
}

/// The variable [`get`] reads first for a provider, e.g. `OCHRE_OPENAI_API_KEY`.
pub fn env_var(provider: &str) -> String {
    format!("OCHRE_{}_API_KEY", canonical(provider).to_uppercase())
}

/// Why a key could not be saved, with the way out. On Linux the usual cause is that no Secret
/// Service (GNOME Keyring, KWallet, KeePassXC) is running or it refused to unlock.
pub fn store_failed(provider: &str, detail: impl std::fmt::Display) -> crate::Error {
    let var = env_var(provider);
    crate::Error::Other(if cfg!(target_os = "linux") {
        format!(
            "Couldn't save the key: no keyring is available ({detail}). Start GNOME Keyring or KWallet (or another Secret Service), or set {var} in your environment and restart Ochre."
        )
    } else {
        format!(
            "Couldn't save the key in the system keychain ({detail}). You can set {var} in your environment instead."
        )
    })
}

/// Save a key to the OS keyring; `None` or empty deletes it.
pub fn store(provider: &str, key: Option<&str>) -> crate::Result<()> {
    let entry = entry(provider).ok_or_else(|| store_failed(provider, "keyring unavailable"))?;
    match key.filter(|k| !k.is_empty()) {
        Some(k) => entry.set_password(k).map_err(|e| store_failed(provider, e)),
        None => {
            let _ = entry.delete_credential();
            // also drop the pre-rename copy, or `get` would resurrect it through the fallback
            if let Some(old) = legacy_entry(provider) {
                let _ = old.delete_credential();
            }
            Ok(())
        }
    }
}

pub fn has(provider: &str) -> bool {
    get(provider).is_some()
}

/// Presence map for the UI (never the keys themselves). Aliases are listed too, so the UI can look
/// a key up by engine id.
pub fn presence() -> std::collections::BTreeMap<String, bool> {
    let mut map: std::collections::BTreeMap<String, bool> = PROVIDERS
        .iter()
        .map(|(p, _)| ((*p).to_string(), has(p)))
        .collect();
    for (alias, target) in ALIASES {
        let v = map.get(*target).copied().unwrap_or(false);
        map.insert((*alias).to_string(), v);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_failure_names_the_env_var() {
        assert_eq!(env_var("google"), "OCHRE_GEMINI_API_KEY");
        let msg = store_failed("openai", "no secret service").to_string();
        assert!(msg.contains("OCHRE_OPENAI_API_KEY") && msg.contains("no secret service"));
    }

    #[test]
    fn google_shares_the_gemini_key() {
        assert_eq!(canonical("google"), "gemini");
        assert_eq!(canonical("openai"), "openai");
        let p = presence();
        assert_eq!(p.get("google"), p.get("gemini"));
    }
}
