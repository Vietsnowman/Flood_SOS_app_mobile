$backendDir = "C:\Flood_SOS_app_mobile\Floodsos1\Sos-backend"
$gopAppDir = "C:\Flood_SOS_app_mobile\Gop_app"

Write-Host "Starting Node.js Backend..."
Start-Process powershell -ArgumentList "-NoExit", "-WindowStyle", "Normal", "-Command", "`$host.UI.RawUI.WindowTitle='Node.js Backend'; cd '$backendDir'; node server.js"

Write-Host "Starting Priority API..."
Start-Process powershell -ArgumentList "-NoExit", "-WindowStyle", "Normal", "-Command", "`$host.UI.RawUI.WindowTitle='Priority API'; cd '$gopAppDir'; uv run uvicorn priority_api:app --host 0.0.0.0 --port 8765 --reload"

Write-Host "Starting Routing API..."
Start-Process powershell -ArgumentList "-NoExit", "-WindowStyle", "Normal", "-Command", "`$host.UI.RawUI.WindowTitle='Routing API'; cd '$gopAppDir'; uv run uvicorn routing_api:app --host 0.0.0.0 --port 8766 --reload"

Write-Host "Starting Streamlit Dashboard (app6.py)..."
Start-Process powershell -ArgumentList "-NoExit", "-WindowStyle", "Normal", "-Command", "`$host.UI.RawUI.WindowTitle='Streamlit Dashboard'; cd '$gopAppDir'; uv run streamlit run app6.py"

Write-Host "Starting Streamlit SOS Map (app_SOS_shelters7.py)..."
Start-Process powershell -ArgumentList "-NoExit", "-WindowStyle", "Normal", "-Command", "`$host.UI.RawUI.WindowTitle='Streamlit SOS Map'; cd '$gopAppDir'; uv run streamlit run app_SOS_shelters7.py"

Write-Host "Đã khởi chạy 5 dịch vụ trong các cửa sổ mới!"
