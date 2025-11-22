// Khởi tạo bản đồ
const map = L.map("map").setView([10.78, 106.70], 13);

// Layer nền
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
}).addTo(map);

// Icon mặc định
const defaultIcon = L.icon({
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
});

// Biến chung
let markerA = null;
let markerB = null;
let currentLine = null;
let distanceLabel = null;
