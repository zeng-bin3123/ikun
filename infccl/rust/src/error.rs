use std::fmt;

#[derive(Debug, Clone)]
pub enum InfcclError {
    Cuda(i32),
    InvalidArg,
    NotFound,
    Timeout,
    TransportFail,
    BatchInvalid,
    InitFailed(String),
    Other(i32),
}

impl InfcclError {
    pub fn from_rc(rc: i32) -> Option<Self> {
        match rc {
            0 => None,
            -1 => Some(Self::Cuda(rc)),
            -2 => Some(Self::Cuda(rc)),
            -3 => Some(Self::InvalidArg),
            -4 => Some(Self::TransportFail),
            -6 => Some(Self::NotFound),
            -8 => Some(Self::Timeout),
            -10 => Some(Self::TransportFail),
            _ => Some(Self::Other(rc)),
        }
    }
}

impl fmt::Display for InfcclError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(rc) => write!(f, "CUDA error (rc={rc})"),
            Self::InvalidArg => write!(f, "invalid argument"),
            Self::NotFound => write!(f, "not found"),
            Self::Timeout => write!(f, "timeout"),
            Self::TransportFail => write!(f, "transport failure"),
            Self::BatchInvalid => write!(f, "invalid batch"),
            Self::InitFailed(s) => write!(f, "init failed: {s}"),
            Self::Other(rc) => write!(f, "error (rc={rc})"),
        }
    }
}

impl std::error::Error for InfcclError {}

pub type Result<T> = std::result::Result<T, InfcclError>;

#[inline]
pub fn check(rc: i32) -> Result<()> {
    match InfcclError::from_rc(rc) {
        None => Ok(()),
        Some(e) => Err(e),
    }
}
