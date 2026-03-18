mod sidecar;

use sidecar::SidecarManager;
use std::sync::Mutex;
use tauri::{AppHandle, Manager};

#[tauri::command]
async fn api_request(
    method: String,
    path: String,
    body: Option<serde_json::Value>,
    app: AppHandle,
) -> Result<serde_json::Value, String> {
    let port = sidecar::get_port(&app).ok_or("Sidecar not running")?;
    let url = format!("http://127.0.0.1:{}{}", port, path);

    let client = reqwest::Client::new();
    let req = match method.to_uppercase().as_str() {
        "GET" => client.get(&url),
        "POST" => {
            let r = client.post(&url);
            if let Some(b) = body { r.json(&b) } else { r }
        }
        "PUT" => {
            let r = client.put(&url);
            if let Some(b) = body { r.json(&b) } else { r }
        }
        "DELETE" => client.delete(&url),
        other => return Err(format!("Unsupported method: {}", other)),
    };

    let resp = req.send().await.map_err(|e| e.to_string())?;
    let status = resp.status();
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    if !status.is_success() {
        return Err(serde_json::to_string(&json).unwrap_or_default());
    }
    Ok(json)
}

#[tauri::command]
async fn start_sidecar(app: AppHandle) -> Result<u16, String> {
    sidecar::spawn_sidecar(&app).await
}

#[tauri::command]
fn get_sidecar_port(app: AppHandle) -> Option<u16> {
    sidecar::get_port(&app)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(SidecarManager(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            api_request,
            start_sidecar,
            get_sidecar_port,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                sidecar::kill_sidecar(window.app_handle());
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
