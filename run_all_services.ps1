$backendDir = "D:\Flood_SOS_app_mobile\Floodsos1\Sos-backend"
$gopAppDir = "D:\Flood_SOS_app_mobile\Gop_app"

Write-Host "Starting Node.js Backend..."
Start-Process powershell -ArgumentList "-Title 'Node.js Backend'", "-NoExit", "-Command", "cd '$backendDir'; node server.js"

Write-Host "Starting Priority API..."
Start-Process powershell -ArgumentList "-Title 'Priority API'", "-NoExit", "-Command", "cd '$gopAppDir'; uv run uvicorn priority_api:app --host 0.0.0.0 --port 8765 --reload"

Write-Host "Starting Routing API..."
Start-Process powershell -ArgumentList "-Title 'Routing API'", "-NoExit", "-Command", "cd '$gopAppDir'; uv run uvicorn routing_api:app --host 0.0.0.0 --port 8766 --reload"

Write-Host "Starting Streamlit Dashboard (app6.py)..."
Start-Process powershell -ArgumentList "-Title 'Streamlit Dashboard'", "-NoExit", "-Command", "cd '$gopAppDir'; uv run streamlit run app6.py"

Write-Host "Starting Streamlit SOS Map (app_SOS_shelters7.py)..."
Start-Process powershell -ArgumentList "-Title 'Streamlit SOS Map'", "-NoExit", "-Command", "cd '$gopAppDir'; uv run streamlit run app_SOS_shelters7.py"

Write-Host "Đã khởi chạy 5 dịch vụ trong các cửa sổ mới!"
