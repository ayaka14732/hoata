use crate::{AppState, CompileRequest};
use serde_json::{Value, json};
use std::os::unix::process::CommandExt;
use std::{process::Stdio, time::Instant};
use tokio::{io::AsyncReadExt, process::Command, sync::watch};

const OUTPUT_LIMIT: u64 = 8 * 1024 * 1024;
const LOG_LIMIT: usize = 128 * 1024;
type Failure = (&'static str, String);

// Own the whole process group, including native bridge compiler subprocesses.
struct ProcessGroup(i32);
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        // SAFETY: the child starts a fresh process group; negative PID addresses it.
        unsafe {
            libc::kill(-self.0, libc::SIGKILL);
        }
    }
}

async fn read_log(mut stream: impl tokio::io::AsyncRead + Unpin) -> String {
    let mut saved = Vec::new();
    let mut buffer = [0_u8; 4096];
    let mut truncated = false;
    while let Ok(count) = stream.read(&mut buffer).await {
        if count == 0 {
            break;
        }
        let remaining = LOG_LIMIT.saturating_sub(saved.len());
        saved.extend_from_slice(&buffer[..count.min(remaining)]);
        truncated |= count > remaining;
    }
    let mut text = String::from_utf8_lossy(&saved).into_owned();
    if truncated {
        text.push_str("\n[Log truncated at 128 KiB]\n");
    }
    text
}

pub(super) async fn compile(
    state: &AppState,
    request: &CompileRequest,
    mut cancelled: watch::Receiver<bool>,
) -> Result<Value, Failure> {
    let directory = tempfile::Builder::new()
        .prefix("hoata-")
        .tempdir()
        .map_err(internal)?;
    let source = directory.path().join("kernel.py");
    let output = directory.path().join("result.json");
    tokio::fs::write(&source, &request.source)
        .await
        .map_err(internal)?;
    let started = Instant::now();
    let mut command = Command::new(&state.python);
    command
        .arg(state.root.join("python/runner.py"))
        .arg(&source)
        .arg(&output)
        .current_dir(directory.path())
        .env("PYTHONPATH", state.root.join("python"))
        .env("JAX_PLATFORMS", "cpu")
        .env("HOATA_TARGET", &request.target)
        .env("ALLOW_MULTIPLE_LIBTPU_LOAD", "true")
        .env("TPU_SKIP_MDS_QUERY", "1")
        .env("TMPDIR", directory.path())
        .env("OMP_NUM_THREADS", "2")
        .env("OPENBLAS_NUM_THREADS", "2")
        .env_remove("LIBTPU_INIT_ARGS")
        .env_remove("XLA_FLAGS")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    command.as_std_mut().process_group(0);
    // Limits avoid accidental runaway local experiments. This is not a sandbox.
    unsafe {
        command.pre_exec(|| {
            let file_limit = libc::rlimit {
                rlim_cur: 16 * 1024 * 1024,
                rlim_max: 16 * 1024 * 1024,
            };
            let core_limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &file_limit) != 0
                || libc::setrlimit(libc::RLIMIT_CORE, &core_limit) != 0
            {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command.spawn().map_err(internal)?;
    let group = ProcessGroup(
        child
            .id()
            .ok_or_else(|| ("runner", "Compiler process did not start.".into()))? as i32,
    );
    let stdout = tokio::spawn(read_log(child.stdout.take().unwrap()));
    let stderr = tokio::spawn(read_log(child.stderr.take().unwrap()));
    let outcome = tokio::select! {
        status = child.wait() => status.map_err(internal),
        _ = tokio::time::sleep(state.timeout) => Err(("timeout", format!("Compilation exceeded {} seconds.", state.timeout.as_secs()))),
        _ = cancelled.changed() => Err(("cancelled", "Compilation cancelled.".into())),
    };
    drop(group);
    if outcome.is_err() {
        let _ = child.wait().await;
    }
    let logs = json!({"stdout":stdout.await.unwrap_or_default(),"stderr":stderr.await.unwrap_or_default()});
    let status = outcome?;
    if !status.success() {
        return Err((
            "process",
            format!(
                "Compiler process exited with {status}.\n{}",
                logs["stderr"].as_str().unwrap_or_default()
            ),
        ));
    }
    if tokio::fs::metadata(&output).await.map_err(internal)?.len() > OUTPUT_LIMIT {
        return Err((
            "output_limit",
            "Compilation output exceeded 8 MiB. Try a smaller kernel.".into(),
        ));
    }
    let bytes = tokio::fs::read(output).await.map_err(internal)?;
    let mut result: Value = serde_json::from_slice(&bytes).map_err(internal)?;
    result["id"] = json!(request.id);
    result["target"] = json!(request.target);
    result["elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
    result["logs"] = logs;
    Ok(result)
}

fn internal(error: impl std::fmt::Display) -> Failure {
    ("runner", error.to_string())
}
