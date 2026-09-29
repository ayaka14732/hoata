mod runner;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::sync::{Mutex, Semaphore, watch};
use tower_http::services::ServeDir;

type ActiveJob = Option<(String, watch::Sender<bool>)>;

struct AppState {
    root: PathBuf,
    python: PathBuf,
    versions: Value,
    semaphore: Arc<Semaphore>,
    active: Mutex<ActiveJob>,
    timeout: Duration,
    port: u16,
}

#[derive(Deserialize)]
struct CompileRequest {
    id: String,
    source: String,
    target: String,
}

fn error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"ok":false,"error":code,"diagnostics":[{"message":message}],"programs":[]})),
    )
        .into_response()
}

async fn local_request(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let local_authority = |value: &str| {
        ["localhost", "127.0.0.1", "[::1]"].iter().any(|host| {
            value == format!("{host}:{}", state.port) || value == format!("{host}:5173")
        })
    };
    if !request
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .is_some_and(local_authority)
    {
        return error(
            StatusCode::FORBIDDEN,
            "local_only",
            "Hoata accepts local requests only.",
        );
    }
    if let Some(origin) = request.headers().get(header::ORIGIN) {
        let allowed = origin
            .to_str()
            .ok()
            .and_then(|value| value.strip_prefix("http://"))
            .is_some_and(local_authority);
        if !allowed {
            return error(
                StatusCode::FORBIDDEN,
                "origin",
                "This origin cannot submit local compilation jobs.",
            );
        }
    }
    next.run(request).await
}

async fn config(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "name":"Hoata", "versions":state.versions,
        "targets":[{"id":"tpu-v4-tc","name":"TPU v4 · TensorCore"},{"id":"tpu-v6e-tc","name":"TPU v6e · TensorCore"}],
        "examples":[
            {"id":"clamp","source":include_str!("../../../examples/clamp.py")},
            {"id":"square","source":include_str!("../../../examples/square.py")},
            {"id":"matmul","source":include_str!("../../../examples/matmul.py")}
        ],
        "timeout_seconds":state.timeout.as_secs(), "source_limit":262144,
    }))
}

async fn compile(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompileRequest>,
) -> Response {
    if !matches!(request.target.as_str(), "tpu-v4-tc" | "tpu-v6e-tc") {
        return error(StatusCode::BAD_REQUEST, "target", "Unsupported TPU target.");
    }
    if request.source.is_empty()
        || request.source.len() > 262_144
        || request.id.len() > 80
        || request.id.is_empty()
    {
        return error(
            StatusCode::BAD_REQUEST,
            "request",
            "Provide a nonempty source of at most 256 KiB and a request ID.",
        );
    }
    let Ok(_permit) = state.semaphore.clone().try_acquire_owned() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "busy",
            "A compilation is already running. Try again after it finishes.",
        );
    };
    let (cancel, cancelled) = watch::channel(false);
    *state.active.lock().await = Some((request.id.clone(), cancel));
    let result = runner::compile(&state, &request, cancelled).await;
    *state.active.lock().await = None;
    match result {
        Ok(value) => Json(value).into_response(),
        Err((code, message)) => error(StatusCode::UNPROCESSABLE_ENTITY, code, &message),
    }
}

async fn cancel(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> StatusCode {
    if let Some((active_id, signal)) = state.active.lock().await.as_ref()
        && active_id == &id
    {
        let _ = signal.send(true);
        return StatusCode::ACCEPTED;
    }
    StatusCode::NOT_FOUND
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let python = std::env::var_os("HOATA_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/srv/workspace/venv/bin/python"));
    let port = std::env::var("HOATA_PORT")
        .unwrap_or_else(|_| "8000".into())
        .parse()?;
    let timeout = Duration::from_secs(
        std::env::var("HOATA_TIMEOUT_SECONDS")
            .unwrap_or_else(|_| "120".into())
            .parse()?,
    );
    let info = tokio::process::Command::new(&python)
        .arg(root.join("python/runner.py"))
        .arg("--info")
        .output()
        .await?;
    if !info.status.success() {
        return Err(format!(
            "Python toolchain unavailable: {}",
            String::from_utf8_lossy(&info.stderr)
        )
        .into());
    }
    let state = Arc::new(AppState {
        root: root.clone(),
        python,
        versions: serde_json::from_slice(&info.stdout)?,
        semaphore: Arc::new(Semaphore::new(1)),
        active: Mutex::new(None),
        timeout,
        port,
    });
    let app = Router::new()
        .route("/api/config", get(config))
        .route("/api/compile", post(compile))
        .route("/api/compile/{id}", axum::routing::delete(cancel))
        .fallback_service(ServeDir::new(root.join("dist")))
        .layer(DefaultBodyLimit::max(1_600_000))
        .layer(middleware::from_fn_with_state(state.clone(), local_request))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await?;
    println!("Hoata · http://127.0.0.1:{port} · local offline compilation");
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {},
                _ = terminate.recv() => {},
            }
            if let Some((_, signal)) = state.active.lock().await.as_ref() {
                let _ = signal.send(true);
            }
        })
        .await?;
    Ok(())
}
