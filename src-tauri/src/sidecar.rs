use std::sync::Mutex;
use std::time::Duration;
use tauri::AppHandle;

pub struct SidecarState {
    pub port: u16,
    pub child_pid: Option<u32>,
}

pub struct SidecarManager(pub Mutex<Option<SidecarState>>);

async fn wait_for_health(port: u16, timeout_secs: u64) -> bool {
    let url = format!("http://127.0.0.1:{}/health", port);
    let deadline = std::time::Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        if std::time::Instant::now() > deadline {
            return false;
        }
        match reqwest::get(&url).await {
            Ok(resp) if resp.status().is_success() => return true,
            _ => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
}

/// In dev mode: attach to an already-running Python sidecar.
/// Port is read from MEMORA_PORT env var, defaulting to 8765.
#[cfg(dev)]
pub async fn spawn_sidecar(app: &AppHandle) -> Result<u16, String> {
    use tauri::Manager;
    let port: u16 = std::env::var("MEMORA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8765);

    log::info!("Dev mode: attaching to sidecar on port {}", port);

    if !wait_for_health(port, 10).await {
        return Err(format!(
            "No sidecar responding on port {}. Start it with: uvicorn main:app --port {}",
            port, port
        ));
    }

    let state = app.state::<SidecarManager>();
    *state.0.lock().unwrap() = Some(SidecarState { port, child_pid: None });

    log::info!("Attached to sidecar on port {}", port);
    Ok(port)
}

/// In release mode: spawn the bundled PyInstaller binary.
#[cfg(not(dev))]
pub async fn spawn_sidecar(app: &AppHandle) -> Result<u16, String> {
    use tauri::Manager;
    use tauri_plugin_shell::ShellExt;

    let port = portpicker::pick_unused_port().expect("no free ports");
    log::info!("Release mode: spawning sidecar on port {}", port);

    let shell = app.shell();
    let (_rx, child) = shell
        .sidecar("memora-sidecar")
        .map_err(|e| e.to_string())?
        .args(["--port", &port.to_string()])
        .spawn()
        .map_err(|e| e.to_string())?;

    let pid = child.pid();

    let state = app.state::<SidecarManager>();
    *state.0.lock().unwrap() = Some(SidecarState { port, child_pid: Some(pid) });

    if !wait_for_health(port, 60).await {
        return Err(format!("Sidecar did not become healthy on port {}", port));
    }

    log::info!("Sidecar healthy on port {}", port);
    Ok(port)
}

pub fn kill_sidecar(app: &AppHandle) {
    use tauri::Manager;
    let state = app.state::<SidecarManager>();
    let mut guard = state.0.lock().unwrap();
    if let Some(s) = guard.take() {
        if let Some(pid) = s.child_pid {
            #[cfg(unix)]
            unsafe { libc::kill(pid as i32, libc::SIGTERM); }
            #[cfg(windows)]
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/F"])
                .status();
            log::info!("Sent kill to sidecar pid {}", pid);
        }
    }
}

pub fn get_port(app: &AppHandle) -> Option<u16> {
    use tauri::Manager;
    app.state::<SidecarManager>().0.lock().unwrap().as_ref().map(|s| s.port)
}
