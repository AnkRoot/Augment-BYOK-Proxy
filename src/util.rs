use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_ms() -> u64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_millis() as u64
}

pub fn normalize_raw_token(token: &str) -> String {
  let mut t = token.trim();
  if t.is_empty() {
    return String::new();
  }

  let lower = t.to_ascii_lowercase();
  if lower.starts_with("bearer ") {
    t = t[7..].trim();
  }

  if let Some((k, v)) = t.split_once('=') {
    let k = k.trim();
    let v = v.trim();
    let looks_like_env = !k.is_empty()
      && !v.is_empty()
      && k
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
      && (k.ends_with("_TOKEN")
        || k.ends_with("_API_TOKEN")
        || k.ends_with("_KEY")
        || k.ends_with("_API_KEY"));
    if looks_like_env {
      t = v;
    }
  }

  t.to_string()
}

pub fn join_url(base_url: &str, endpoint: &str) -> anyhow::Result<String> {
  let mut base = base_url.trim().to_string();
  if !base.ends_with('/') {
    base.push('/');
  }
  let endpoint = endpoint.trim_start_matches('/');
  let url = format!("{base}{endpoint}");
  let _ = url::Url::parse(&url)?;
  Ok(url)
}
