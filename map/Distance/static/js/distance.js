// ==============================
//  Tính khoảng cách A → B
// ==============================
async function calcDistance(a = null, b = null) {
    // Nếu bị truyền event → bỏ
    if (a instanceof Event) a = null;
    if (b instanceof Event) b = null;

    // Không có tham số → lấy từ input
    if (!a || !b) {
        a = document.getElementById("start").value.trim();
        b = document.getElementById("end").value.trim();
    }

    if (!a || !b) {
        alert("Vui lòng nhập A và B");
        return;
    }

    try {
        const res = await fetch(`/api/distance?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);

        if (!res.ok) {
            alert("Không tìm thấy vị trí! Kiểm tra lại địa chỉ.");
            return;
        }

        const data = await res.json();
        showRouteOnMap(data);

    } catch (err) {
        alert("Lỗi kết nối đến server!");
        console.error(err);
    }
}

// ==============================
//  Tính khoảng cách vị trí hiện tại → B
// ==============================
async function calcFromHere(dest = null) {

    // Nếu bị truyền event → bỏ
    if (dest instanceof Event) dest = null;

    if (!dest)
        dest = document.getElementById("end").value.trim();

    if (!dest) {
        alert("Nhập điểm B trước!");
        return;
    }

    navigator.geolocation.getCurrentPosition(async (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;

        const startString = `${lat}, ${lon}`;

        await calcDistance(startString, dest);
    });
}



// ==============================
//  Hiển thị đường đi & marker
// ==============================
function showRouteOnMap(data) {
    const { from, to, distance_km } = data;

    // Xóa mọi layer cũ
    if (markerA) map.removeLayer(markerA);
    if (markerB) map.removeLayer(markerB);
    if (currentLine) map.removeLayer(currentLine);
    if (distanceLabel) map.removeLayer(distanceLabel);

    // Marker A
    markerA = L.marker([from.lat, from.lon], { icon: defaultIcon })
        .addTo(map)
        .bindPopup("Điểm A")
        .openPopup();

    // Marker B
    markerB = L.marker([to.lat, to.lon], { icon: defaultIcon })
        .addTo(map)
        .bindPopup("Điểm B");

    // Vẽ đường thẳng A → B
    currentLine = L.polyline(
        [
            [from.lat, from.lon],
            [to.lat, to.lon]
        ],
        { color: "blue", weight: 4 }
    ).addTo(map);

    map.fitBounds(currentLine.getBounds(), { padding: [40, 40] });

    // Nhãn hiển thị số km
    const midLat = (from.lat + to.lat) / 2;
    const midLon = (from.lon + to.lon) / 2;

    distanceLabel = L.marker([midLat, midLon], {
        icon: L.divIcon({
            className: "distance-label",
            html: `<div class="distance-box">${distance_km} km</div>`
        })
    }).addTo(map);
}
