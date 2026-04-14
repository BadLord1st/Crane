use std::fmt;
use std::str::FromStr;

use anyhow::{anyhow, Result};

const SUPPORTED_KV_CACHE_MODES: &[&str] = &["bf16_dense", "int8_rowwise_kv", "turboquant"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheMode {
    #[default]
    Bf16Dense,
    Int8RowwiseKv,
    TurboQuant,
}

impl KvCacheMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16Dense => "bf16_dense",
            Self::Int8RowwiseKv => "int8_rowwise_kv",
            Self::TurboQuant => "turboquant",
        }
    }

    fn parse_err(value: &str) -> anyhow::Error {
        anyhow!(
            "Unsupported KV cache mode '{value}'. Supported modes: {}",
            SUPPORTED_KV_CACHE_MODES.join("|")
        )
    }

    pub fn resolve(cli_value: Option<&str>, env_value: Option<&str>) -> Result<Self> {
        match cli_value.or(env_value) {
            Some(value) => Self::from_str(value),
            None => Ok(Self::default()),
        }
    }
}

impl fmt::Display for KvCacheMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for KvCacheMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bf16_dense" => Ok(Self::Bf16Dense),
            "int8_rowwise_kv" => Ok(Self::Int8RowwiseKv),
            "turboquant" => Ok(Self::TurboQuant),
            other => Err(Self::parse_err(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvBackendConfig {
    pub mode: KvCacheMode,
}

impl KvBackendConfig {
    pub fn resolve(cli_value: Option<&str>, env_value: Option<&str>) -> Result<Self> {
        Ok(Self {
            mode: KvCacheMode::resolve(cli_value, env_value)?,
        })
    }
}
