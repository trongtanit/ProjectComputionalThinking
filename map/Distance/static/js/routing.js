// =======================
//  HÀM TÌM ĐƯỜNG A → B
// =======================

async function findRoute(start = null, end = null) {

    // Nếu nhận vào sự kiện click → reset
    if (start instanceof Event) start = null;
    if (end instanceof Event) end = null;

    // Nếu không truyền tham số thì lấy từ input
    if (!start || !end) {
        start = document.getElementById("start").value.trim();
        end = document.getElementById("end").value.trim();
    }

    // Xóa khoảng trắng thừa hoặc dấu phẩy sai định dạng
    if (typeof start === "string") start = start.replace(/\s+/g, "");
    if (typeof end === "string") end = end.replace(/\s+/g, "");

    // Kiểm tra input
    if (!start || !end || !start.includes(",") || !end.includes(",")) {
        alert("Vui lòng nhập tọa độ theo dạng: lat,lng");
        return;
    }

    try {
        const res = await fetch(
            `/api/route?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`
        );

        if (!res.ok) {
            const msg = await res.text();
            console.error("SERVER ERROR:", msg);
            alert("Không tìm được đường đi!");
            return;
        }

        const data = await res.json();

        if (!data.geometry || !Array.isArray(data.geometry)) {
            console.log("DỮ LIỆU TRẢ VỀ:", data);
            alert("API trả về dữ liệu không hợp lệ!");
            return;
        }

        drawRoute(data);

    } catch (err) {
        alert("Lỗi kết nối server!");
        console.error("FETCH ERROR:", err);
    }
}

// =============================
// TÌM ĐƯỜNG TỪ VỊ TRÍ HIỆN TẠI
// =============================

async function findRouteFromHere(dest = null) {

    // Tránh PointerEvent
    if (dest instanceof Event) dest = null;

    if (!dest) {
        dest = document.getElementById("end").value.trim();
    }

    if (!dest) {
        alert("Nhập điểm B trước!");
        return;
    }

    navigator.geolocation.getCurrentPosition(async (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;

        const startString = `${lat}, ${lon}`;

        await findRoute(startString, dest);
    });
}


// =======================
//     VẼ ĐƯỜNG ĐI
// =======================

let routeLine = null;
let routeLabel = null;

function drawRoute(data) {
    const coords = data.geometry.map(c => [c[1], c[0]]);

    if (routeLine) map.removeLayer(routeLine);
    if (routeLabel) map.removeLayer(routeLabel);

    routeLine = L.polyline(coords, { color: "red", weight: 5 }).addTo(map);

    map.fitBounds(routeLine.getBounds(), { padding: [40, 40] });

    const midPoint = coords[Math.floor(coords.length / 2)];

    routeLabel = L.marker(midPoint, {
        icon: L.divIcon({
            html: `<div class="distance-box">${data.distance_km} km – ${data.duration_min} phút</div>`
        })
    }).addTo(map);
}

