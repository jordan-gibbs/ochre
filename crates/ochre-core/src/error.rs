use std::time::Duration;

/// One error type across crates. `code` strings are stable: the UI maps them to messages.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Config(String),
    #[error("missing API key for {0}")]
    MissingKey(String),
    #[error("{provider}: authentication failed")]
    Auth { provider: String },
    #[error("{provider}: rate limited or out of quota")]
    Quota { provider: String },
    #[error("{provider}: network error: {message}")]
    Network { provider: String, message: String },
    #[error("timed out after {0:?}")]
    Timeout(Duration),
    #[error("model: {0}")]
    Model(String),
    #[error("download failed: {0}")]
    Download(String),
    #[error("audio: {0}")]
    Audio(String),
    #[error("could not type into the focused window: {0}")]
    Inject(String),
    #[error("permission needed: {0}")]
    Permission(String),
    #[error("canceled")]
    Canceled,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn code(&self) -> &'static str {
        match self {
            Error::Config(_) => "config",
            Error::MissingKey(_) => "missing_key",
            Error::Auth { .. } => "auth",
            Error::Quota { .. } => "quota",
            Error::Network { .. } => "network",
            Error::Timeout(_) => "timeout",
            Error::Model(_) => "model",
            Error::Download(_) => "download",
            Error::Audio(_) => "audio",
            Error::Inject(_) => "inject",
            Error::Permission(_) => "permission",
            Error::Canceled => "canceled",
            Error::Io(_) => "io",
            Error::Other(_) => "other",
        }
    }

    /// Failures a cloud stage may fall back from (to the local engine / raw text).
    pub fn is_fallback_worthy(&self) -> bool {
        matches!(
            self,
            Error::Network { .. }
                | Error::Timeout(_)
                | Error::Quota { .. }
                | Error::Auth { .. }
                | Error::MissingKey(_)
        )
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
