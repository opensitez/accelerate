use std::time::Duration;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    LayerPresent,
    ImageCreate,
    Upload,
    Commit,
}

/// Optional host profiling hooks, without a dependency on a browser or logger.
#[derive(Clone, Copy)]
pub struct Diagnostics {
    pub enabled: fn() -> bool,
    pub media_enabled: fn() -> bool,
    pub record: fn(Phase, Duration),
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self {
            enabled: || false,
            media_enabled: || false,
            record: |_, _| {},
        }
    }
}

impl Diagnostics {
    pub(crate) fn span(self, phase: Phase) -> Span {
        Span {
            phase,
            started: (self.enabled)().then(Instant::now),
            record: self.record,
        }
    }
}

pub(crate) struct Span {
    phase: Phase,
    started: Option<Instant>,
    record: fn(Phase, Duration),
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            (self.record)(self.phase, started.elapsed());
        }
    }
}
