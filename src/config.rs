use std::{collections::BTreeMap, fs, net::SocketAddr, path::Path};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use url::Url;

fn default_logging_filter() -> String {
  "info".to_string()
}

fn default_thinking_enabled() -> bool {
  true
}

fn default_thinking_budget_tokens() -> u32 {
  10000
}

fn default_max_tokens() -> u32 {
  8192
}

fn default_timeout_seconds() -> u64 {
  120
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
  pub server: ServerConfig,
  pub proxy: ProxyConfig,
  pub official: OfficialConfig,
  pub byok: ByokConfig,
  #[serde(default)]
  pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
  pub host: String,
  pub port: u16,
}

impl ServerConfig {
  pub fn socket_addr(&self) -> anyhow::Result<SocketAddr> {
    let addr: SocketAddr = format!("{}:{}", self.host, self.port)
      .parse()
      .context("server.host/server.port 不是合法 socket 地址")?;
    Ok(addr)
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProxyConfig {
  pub auth_token: String,
}

impl ProxyConfig {
  pub fn validate(&self) -> anyhow::Result<()> {
    if self.auth_token.trim().is_empty() {
      anyhow::bail!("proxy.auth_token 不能为空");
    }
    Ok(())
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OfficialConfig {
  pub base_url: String,
  pub api_token: String,
}

impl OfficialConfig {
  pub fn validate(&self) -> anyhow::Result<()> {
    if self.base_url.trim().is_empty() {
      anyhow::bail!("official.base_url 不能为空");
    }
    if self.api_token.trim().is_empty() {
      anyhow::bail!("official.api_token 不能为空");
    }
    let _ = Url::parse(&self.base_url).context("official.base_url 不是合法 URL")?;
    Ok(())
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ByokConfig {
  pub providers: Vec<ProviderConfig>,
  #[serde(default)]
  pub active_provider_id: Option<String>,
}

impl ByokConfig {
  pub fn validate(&self) -> anyhow::Result<()> {
    if self.providers.is_empty() {
      anyhow::bail!("byok.providers 不能为空");
    }

    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &self.providers {
      p.validate()?;
      let id = p.id().trim();
      if id.is_empty() {
        anyhow::bail!("byok.providers[].id 不能为空");
      }
      if id.contains(':') {
        anyhow::bail!(
          "byok.providers[].id 不能包含 ':'（会破坏 byok:<providerId>:<modelId> 解析）：{id}"
        );
      }
      if !seen_ids.insert(id.to_string()) {
        anyhow::bail!("byok.providers[].id 重复：{id}");
      }
    }
    if let Some(id) = &self.active_provider_id {
      let id = id.trim();
      if id.is_empty() {
        anyhow::bail!("byok.active_provider_id 不能为空字符串（或删除该字段）");
      }
      if !self.providers.iter().any(|p| p.id().trim() == id) {
        anyhow::bail!("byok.active_provider_id 未命中 providers: {id}");
      }
    }
    Ok(())
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ProviderConfig {
  #[serde(rename = "anthropic")]
  Anthropic(AnthropicProviderConfig),
  #[serde(
    rename = "openai_compatible",
    alias = "openai-compatible",
    alias = "openai"
  )]
  OpenAICompatible(OpenAICompatibleProviderConfig),
}

impl ProviderConfig {
  pub fn id(&self) -> &str {
    match self {
      ProviderConfig::Anthropic(p) => p.id.as_str(),
      ProviderConfig::OpenAICompatible(p) => p.id.as_str(),
    }
  }

  pub fn validate(&self) -> anyhow::Result<()> {
    match self {
      ProviderConfig::Anthropic(p) => p.validate(),
      ProviderConfig::OpenAICompatible(p) => p.validate(),
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicProviderConfig {
  pub id: String,
  pub base_url: String,
  pub api_key: String,
  pub default_model: String,
  #[serde(default = "default_max_tokens")]
  pub max_tokens: u32,
  #[serde(default = "default_timeout_seconds")]
  pub timeout_seconds: u64,
  #[serde(default)]
  pub thinking: ThinkingConfig,
  #[serde(default)]
  pub extra_headers: BTreeMap<String, String>,
}

impl AnthropicProviderConfig {
  pub fn validate(&self) -> anyhow::Result<()> {
    if self.id.trim().is_empty() {
      anyhow::bail!("byok.providers[type=anthropic].id 不能为空");
    }
    if self.base_url.trim().is_empty() {
      anyhow::bail!("byok.providers[type=anthropic].base_url 不能为空");
    }
    if self.api_key.trim().is_empty() {
      anyhow::bail!("byok.providers[type=anthropic].api_key 不能为空");
    }
    if self.default_model.trim().is_empty() {
      anyhow::bail!("byok.providers[type=anthropic].default_model 不能为空");
    }
    let _ =
      Url::parse(&self.base_url).context("byok.providers[type=anthropic].base_url 不是合法 URL")?;
    Ok(())
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAICompatibleProviderConfig {
  pub id: String,
  pub base_url: String,
  pub api_key: String,
  pub default_model: String,
  #[serde(default = "default_max_tokens")]
  pub max_tokens: u32,
  #[serde(default = "default_timeout_seconds")]
  pub timeout_seconds: u64,
  #[serde(default)]
  pub extra_headers: BTreeMap<String, String>,
}

impl OpenAICompatibleProviderConfig {
  pub fn validate(&self) -> anyhow::Result<()> {
    if self.id.trim().is_empty() {
      anyhow::bail!("byok.providers[type=openai_compatible].id 不能为空");
    }
    if self.base_url.trim().is_empty() {
      anyhow::bail!("byok.providers[type=openai_compatible].base_url 不能为空");
    }
    if self.api_key.trim().is_empty() {
      anyhow::bail!("byok.providers[type=openai_compatible].api_key 不能为空");
    }
    if self.default_model.trim().is_empty() {
      anyhow::bail!("byok.providers[type=openai_compatible].default_model 不能为空");
    }
    let _ = Url::parse(&self.base_url)
      .context("byok.providers[type=openai_compatible].base_url 不是合法 URL")?;
    Ok(())
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThinkingConfig {
  #[serde(default = "default_thinking_enabled")]
  pub enabled: bool,
  #[serde(default = "default_thinking_budget_tokens")]
  pub budget_tokens: u32,
}

impl Default for ThinkingConfig {
  fn default() -> Self {
    Self {
      enabled: default_thinking_enabled(),
      budget_tokens: default_thinking_budget_tokens(),
    }
  }
}

impl Config {
  pub fn load(path: &Path) -> anyhow::Result<Self> {
    let bytes = fs::read(path).with_context(|| format!("读取配置失败: {}", path.display()))?;
    let config: Self =
      serde_yaml::from_slice(&bytes).context("解析 YAML 配置失败 (config.yaml)")?;
    config.validate()?;
    Ok(config)
  }

  pub fn save(&self, path: &Path) -> anyhow::Result<()> {
    self.validate()?;
    let yaml = serde_yaml::to_string(self).context("序列化 YAML 配置失败")?;
    fs::write(path, yaml).with_context(|| format!("写入配置失败: {}", path.display()))?;
    Ok(())
  }

  pub fn validate(&self) -> anyhow::Result<()> {
    if self.server.host.trim().is_empty() {
      anyhow::bail!("server.host 不能为空");
    }
    if self.server.port == 0 {
      anyhow::bail!("server.port 不能为 0");
    }
    self.proxy.validate()?;
    self.official.validate()?;
    self.byok.validate()?;
    self.logging.validate()?;
    Ok(())
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
  #[serde(default = "default_logging_filter")]
  pub filter: String,
  #[serde(default)]
  pub dump_chat_stream_body: bool,
}

impl Default for LoggingConfig {
  fn default() -> Self {
    Self {
      filter: default_logging_filter(),
      dump_chat_stream_body: false,
    }
  }
}

impl LoggingConfig {
  pub fn validate(&self) -> anyhow::Result<()> {
    if self.filter.trim().is_empty() {
      anyhow::bail!("logging.filter 不能为空");
    }
    tracing_subscriber::EnvFilter::try_new(self.filter.trim())
      .context("logging.filter 不是合法 tracing filter (EnvFilter 语法)")?;
    Ok(())
  }
}

pub fn init_tracing(logging: &LoggingConfig) -> anyhow::Result<()> {
  let filter = tracing_subscriber::EnvFilter::try_new(logging.filter.trim())
    .context("logging.filter 不是合法 tracing filter (EnvFilter 语法)")?;
  tracing_subscriber::fmt()
    .with_env_filter(filter)
    .with_target(false)
    .compact()
    .init();
  Ok(())
}
