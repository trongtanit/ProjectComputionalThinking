from flask import Flask, render_template
from routes.geocode import geocode_bp
from routes.distance import distance_bp
from routes.route import route_bp

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(geocode_bp, url_prefix="/api")
app.register_blueprint(distance_bp, url_prefix="/api")
app.register_blueprint(route_bp, url_prefix="/api")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
