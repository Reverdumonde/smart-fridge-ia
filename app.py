"""app.py - Smart Fridge Recipe Recommendation System"""

"""
Smart Fridge Recipe Recommendation System
==========================================
IB Computer Science SL Internal Assessment
Author: [Your Name]

This Flask application helps users reduce food waste by recommending recipes
based on the ingredients currently in their fridge and their expiration dates.

The recommendation algorithm is the core of this project (see recommend_recipes()).
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import json
import os
import math
from datetime import datetime, date
import functools

# ─────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smartfridge_ib_secret_key_2024")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(BASE_DIR, "fridge.db")
RECIPES_PATH = os.path.join(BASE_DIR, "recipes.json")

# Max days before we consider an ingredient "fresh" (used to normalise urgency)
MAX_FRESHNESS_DAYS = 14
MATCH_WEIGHT = 0.85
URGENCY_WEIGHT = 0.15


def load_env_file():
    """Load key=value pairs from a local .env file if present."""
    env_paths = [
        os.path.join(PROJECT_DIR, ".env"),
        os.path.join(BASE_DIR, ".env"),
    ]

    for env_path in env_paths:
        if not os.path.exists(env_path):
            continue

        with open(env_path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value


load_env_file()


# ─────────────────────────────────────────
# Database Setup
# ─────────────────────────────────────────
def get_db():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # Rows behave like dicts
    return conn


def init_db():
    """Create tables if they don't already exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ingredients (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER NOT NULL,
                name            TEXT    NOT NULL,
                date_added      TEXT    NOT NULL,
                expiration_date TEXT,               -- NULL means no expiry set
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)


# ─────────────────────────────────────────
# Authentication Helpers
# ─────────────────────────────────────────
def login_required(view):
    """Decorator: redirect to /login if the user is not logged in."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session and not session.get("guest"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# ─────────────────────────────────────────
# Urgency Score Algorithm
# ─────────────────────────────────────────
def compute_days_left(expiration_date_str):
    """
    Calculate how many days remain until an ingredient expires.

    Parameters:
        expiration_date_str (str | None): ISO date string "YYYY-MM-DD" or None

    Returns:
        int | None: Days remaining, or None if no expiry date was provided.
                    Negative values mean the ingredient is already expired.
    """
    if not expiration_date_str:
        return None
    try:
        exp = date.fromisoformat(expiration_date_str)
        return (exp - date.today()).days
    except ValueError:
        return None


def compute_urgency(days_left):
    """
    Convert days_left into a 0–1 urgency score.

    Algorithm:
        urgency = (MAX_FRESHNESS_DAYS - days_left) / MAX_FRESHNESS_DAYS

    Interpretation:
        • days_left = 0  → urgency = 1.0  (expires today → very urgent)
        • days_left = 14 → urgency = 0.0  (still fresh → not urgent)
        • days_left < 0  → urgency clamped to 1.0  (already expired)
        • days_left > 14 → urgency clamped to 0.0  (very fresh)

    Parameters:
        days_left (int | None): Value from compute_days_left()

    Returns:
        float: Urgency score in [0.0, 1.0]
    """
    if days_left is None:
        return 0.0   # No expiry set → treat as low urgency

    raw = (MAX_FRESHNESS_DAYS - days_left) / MAX_FRESHNESS_DAYS
    return max(0.0, min(1.0, raw))   # Clamp to [0, 1]


def urgency_label(urgency_score):
    """
    Convert a numeric urgency score into a human-readable category.

    Thresholds:
        ≥ 0.66 → "high"   (shown in red)
        ≥ 0.33 → "medium" (shown in yellow/orange)
        < 0.33 → "low"    (shown in green)
    """
    if urgency_score >= 0.66:
        return "high"
    elif urgency_score >= 0.33:
        return "medium"
    else:
        return "low"


# ─────────────────────────────────────────
# Recipe Recommendation Algorithm
# ─────────────────────────────────────────
def recommend_recipes(user_ingredients):
    """
    Core recommendation algorithm — NO AI is used here.

    The algorithm scores every recipe in recipes.json using two factors:

        1. match_rate    = (# of recipe ingredients the user HAS) / (total recipe ingredients)
                          Range: 0.0 – 1.0

        2. urgency_score = average urgency of the matched ingredients
                          This prioritises using ingredients that are about to expire.
                          Range: 0.0 – 1.0

        3. final_score   = (match_rate × 0.85) + (urgency_score × 0.15)
                          85 % weight on coverage, 15 % on using expiring items.

    Only recipes with at least one matching ingredient are included.
    Results are sorted by priority so the most makeable recipes appear first.

    Parameters:
        user_ingredients (list[dict]): Each dict has 'name', 'urgency', etc.

    Returns:
        list[dict]: Matching recipe dicts, each augmented with:
            - 'score'          : priority_score (2 d.p.)
            - 'match_rate'     : percentage string e.g. "75%"
            - 'missing'        : list of ingredient names the user lacks
            - 'matched'        : list of ingredient names the user has
            - 'explanation'    : human-readable score breakdown
    """
    # Load recipes from the local JSON file
    with open(RECIPES_PATH, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    # Build a lookup: ingredient name (lower-case) → urgency score
    # This allows O(1) lookups when matching recipe ingredients
    user_lookup = {
        ing["name"].lower(): ing["urgency"]
        for ing in user_ingredients
    }

    scored = []

    for recipe in recipes:
        recipe_ingredients = [r.lower() for r in recipe["ingredients"]]
        total = len(recipe_ingredients)

        if total == 0:
            continue

        # ── Step 1: Find which ingredients the user has ──────────────────────
        matched_names   = []
        matched_urgency = []
        missing_names   = []

        for ing in recipe_ingredients:
            if ing in user_lookup:
                matched_names.append(ing)
                matched_urgency.append(user_lookup[ing])
            else:
                missing_names.append(ing)

        matched_count = len(matched_names)

        # Skip recipes with zero matches (not useful to show)
        if matched_count == 0:
            continue

        # ── Step 2: Calculate match_rate ─────────────────────────────────────
        match_rate = matched_count / total

        # ── Step 3: Calculate urgency_score ──────────────────────────────────
        # Average urgency of matched ingredients.
        # If no urgency data exists (no expiry dates set), defaults to 0.
        urgency_score = (
            sum(matched_urgency) / len(matched_urgency)
            if matched_urgency else 0.0
        )

        # ── Step 4: Calculate final_score ────────────────────────────────────
        # Coverage matters most; urgency only nudges similar recipes upward.
        priority_score = (match_rate * MATCH_WEIGHT) + (urgency_score * URGENCY_WEIGHT)

        # ── Step 5: Build human-readable explanation ──────────────────────────
        match_pct = round(match_rate * 100)
        if urgency_score >= 0.66:
            urg_text = "high urgency ingredients"
        elif urgency_score >= 0.33:
            urg_text = "medium urgency ingredients"
        else:
            urg_text = "low urgency ingredients"

        explanation = f"{matched_count}/{total} ingredients ready ({match_pct}% match) + {urg_text}"

        scored.append({
            **recipe,
            "score":         round(priority_score, 2),
            "match_rate":    f"{match_pct}%",
            "match_ratio":   match_rate,
            "matched_count": matched_count,
            "total_count":   total,
            "urgency_avg":   round(urgency_score, 2),
            "missing":       missing_names,
            "matched":       matched_names,
            "explanation":   explanation,
        })

    # Sort by final priority score (combination of match and urgency).
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored


# ─────────────────────────────────────────
# Helper: fetch and annotate user ingredients
# ─────────────────────────────────────────
def get_user_ingredients(user_id):
    """
    Retrieve all ingredients for a user from the DB, computing urgency fields.

    Returns a list of dicts ready for rendering in templates and for
    passing to recommend_recipes().
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM ingredients WHERE user_id = ? ORDER BY expiration_date ASC",
            (user_id,)
        ).fetchall()

    result = []
    for row in rows:
        days_left = compute_days_left(row["expiration_date"])
        urgency   = compute_urgency(days_left)
        result.append({
            "id":              row["id"],
            "name":            row["name"],
            "date_added":      row["date_added"],
            "expiration_date": row["expiration_date"],
            "days_left":       days_left,
            "urgency":         urgency,
            "urgency_label":   urgency_label(urgency),
        })
    return result


def get_guest_ingredients():
    """Return temporary guest ingredients stored only in the current session."""
    result = []
    for item in session.get("guest_ingredients", []):
        days_left = compute_days_left(item.get("expiration_date"))
        urgency = compute_urgency(days_left)
        result.append({
            **item,
            "days_left": days_left,
            "urgency": urgency,
            "urgency_label": urgency_label(urgency),
        })
    return result


# ─────────────────────────────────────────
# Routes — Authentication
# ─────────────────────────────────────────
@app.route("/")
def index():
    """Landing page — redirect to dashboard if logged in."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration. Stores a hashed password (never plaintext)."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Both fields are required.", "danger")
            return render_template("register.html")

        hashed = generate_password_hash(password)
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, hashed)
                )
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already taken.", "danger")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Validate credentials and start a session."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/guest")
def guest():
    """Start a temporary guest session without requiring registration."""
    session.clear()
    session["guest"] = True
    session["username"] = "Guest"
    session["guest_ingredients"] = []
    flash("Guest mode started. Add ingredients to try the recommendation features.", "info")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    """Clear the session and redirect to the landing page."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ─────────────────────────────────────────
# Routes — Fridge / Ingredient Management
# ─────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    """Main fridge view: show all ingredients with urgency colour coding."""
    is_guest = session.get("guest", False)
    ingredients = (
        get_guest_ingredients()
        if is_guest
        else get_user_ingredients(session["user_id"])
    )
    return render_template(
        "dashboard.html", ingredients=ingredients, is_guest=is_guest
    )


@app.route("/add_ingredient", methods=["POST"])
@login_required
def add_ingredient():
    """Add a new ingredient to the user's fridge."""
    if session.get("guest"):
        name = request.form.get("name", "").strip()
        exp = request.form.get("expiration_date", "").strip() or None

        if not name:
            flash("Ingredient name is required.", "danger")
            return redirect(url_for("dashboard"))

        guest_ingredients = session.get("guest_ingredients", [])
        next_id = max((item["id"] for item in guest_ingredients), default=0) + 1
        guest_ingredients.append({
            "id": next_id,
            "name": name,
            "date_added": date.today().isoformat(),
            "expiration_date": exp,
        })
        session["guest_ingredients"] = guest_ingredients
        flash(f"'{name}' added to your temporary guest fridge!", "success")
        return redirect(url_for("dashboard"))

    name    = request.form.get("name", "").strip()
    exp     = request.form.get("expiration_date", "").strip()

    if not name:
        flash("Ingredient name is required.", "danger")
        return redirect(url_for("dashboard"))

    today = date.today().isoformat()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO ingredients (user_id, name, date_added, expiration_date)
               VALUES (?, ?, ?, ?)""",
            (session["user_id"], name, today, exp or None)
        )

    flash(f"'{name}' added to your fridge!", "success")
    return redirect(url_for("dashboard"))


@app.route("/delete_ingredient/<int:ing_id>", methods=["POST"])
@login_required
def delete_ingredient(ing_id):
    """Remove a single ingredient (only if it belongs to the current user)."""
    if session.get("guest"):
        guest_ingredients = [
            item for item in session.get("guest_ingredients", [])
            if item["id"] != ing_id
        ]
        session["guest_ingredients"] = guest_ingredients
        flash("Ingredient removed from the temporary guest fridge.", "info")
        return redirect(url_for("dashboard"))

    with get_db() as conn:
        conn.execute(
            "DELETE FROM ingredients WHERE id = ? AND user_id = ?",
            (ing_id, session["user_id"])
        )
    flash("Ingredient removed.", "info")
    return redirect(url_for("dashboard"))


# ─────────────────────────────────────────
# Routes — Recipes & Recommendations
# ─────────────────────────────────────────
@app.route("/recommendations")
@login_required
def recommendations():
    """
    Run the recommendation algorithm and display matching recipes.

    This calls recommend_recipes() which implements the core scoring logic.
    No AI or external API is involved in selecting or ranking recipes.
    """
    ingredients = (
        get_guest_ingredients()
        if session.get("guest")
        else get_user_ingredients(session["user_id"])
    )

    if not ingredients:
        flash("Add some ingredients to your fridge first!", "warning")
        return redirect(url_for("dashboard"))

    top_recipes = recommend_recipes(ingredients)
    return render_template(
        "recommendations.html",
        recipes=top_recipes,
        ingredient_count=len(ingredients),
        match_weight=MATCH_WEIGHT,
        urgency_weight=URGENCY_WEIGHT,
        max_freshness_days=MAX_FRESHNESS_DAYS,
    )


@app.route("/cook/<recipe_name>")
@login_required
def cook(recipe_name):
    """
    Display cooking instructions for a specific recipe.

    Optionally uses the Claude / OpenAI API to rewrite steps into
    natural sentences and add cooking tips (see template JS).
    The recommendation logic itself does NOT use any AI.
    """
    with open(RECIPES_PATH, encoding="utf-8") as f:
        all_recipes = json.load(f)

    recipe = next(
        (r for r in all_recipes if r["name"].lower() == recipe_name.lower()),
        None
    )

    if not recipe:
        flash("Recipe not found.", "danger")
        return redirect(url_for("recommendations"))

    return render_template("cook.html", recipe=recipe)


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────
if __name__ == "__main__":
    init_db()  # Create tables on first run
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
