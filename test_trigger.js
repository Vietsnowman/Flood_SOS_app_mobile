const fs = require('fs');

async function triggerSOS() {
    const formData = new FormData();
    formData.append('name', 'Auto Tester');
    formData.append('phone', '0999999999');
    formData.append('lat', '18.666');
    formData.append('lon', '105.666');
    formData.append('message', 'Chạy thử tự động hệ thống routing Python AI!');
    formData.append('water_level', 'Khẩn cấp');
    formData.append('people_count', '5');

    try {
        console.log("Đang bắn tín hiệu SOS lên máy chủ NodeJS...");
        const response = await fetch('http://127.0.0.1:3002/api/sos/voice', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        console.log("Kết quả từ Server:");
        console.log(JSON.stringify(data, null, 2));
        console.log("\\n=> Tín hiệu đã được hệ thống ghim lên mảng Polyline. Bạn hãy mở MapScreen trên app để xem kết quả!");
    } catch (err) {
        console.error("Lỗi:", err);
    }
}

triggerSOS();
