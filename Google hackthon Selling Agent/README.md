# VendorGroove Net

> **AI-powered platform for US street-market vegetable vendors** — geospatial discovery, RL-driven restock planning, LSTM anomaly pricing, GNN neighbourhood analysis, group-buying coordination, and a Gemini-powered AI chat assistant, all in one dark-mode dashboard.

---

## Introduction

Small vegetable vendors at US farmers' markets face a common trio of problems:

| Problem | Pain |
|---|---|
| **Demand uncertainty** | Over-stock spoils; under-stock loses sales |
| **Isolated purchasing** | Each stand buys alone, paying retail truck rates |
| **No market intelligence** | No visibility into what neighbouring stands sell or trend |

**VendorGroove Net** solves all three in a single web app:

1. **Smart Restock** — A Q-Learning agent (1 247 training episodes) recommends tomorrow's order quantity based on inventory change, sales velocity, and weather signal.
2. **Dynamic Pricing** — A 2-layer LSTM detects demand anomalies per SKU and triggers automatic markdowns or halt-sales badges.
3. **Neighbourhood Insights** — A 2-layer Graph Convolutional Network maps the local vendor graph and surfaces trending items, niche positioning, and low-risk trial SKUs.
4. **Group Buying** — Vendors within a 5-mile radius are auto-matched into shared truck manifests, cutting last-mile delivery cost.
5. **Midori AI** — A Gemini 2.0 Flash chat assistant gives plain-English sales tips, embedded in the dashboard alongside a Live2D VTuber avatar.

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser  (port 1234)                        │
│                                                                     │
│  ┌───────────────┐  ┌──────────────────────────────────────────┐   │
│  │  index.html   │  │              dashboard.html               │   │
│  │  (Onboarding) │  │                                          │   │
│  │               │  │  ┌──────────┐ ┌──────────┐ ┌─────────┐  │   │
│  │  • Shop name  │  │  │  Home    │ │ Restock  │ │  Items  │  │   │
│  │  • Geoloc     │  │  │  Midori  │ │  RL tab  │ │  LSTM   │  │   │
│  │  • Leaflet map│  │  │  VTuber  │ │          │ │  GNN    │  │   │
│  └───────┬───────┘  │  └──────────┘ └──────────┘ └─────────┘  │   │
│          │          │  ┌──────────────────────────────────────┐ │   │
│          │          │  │        Group Buying tab               │ │   │
│          │          │  │  Input → Match engine → Map manifest  │ │   │
│          │          │  └──────────────────────────────────────┘ │   │
│          │          └──────────────────────────────────────────┘   │
└──────────┼──────────────────────────────────────────────────────────┘
           │  REST / JSON
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI  (Uvicorn ASGI)                         │
│                                                                     │
│  /api/vendors          /api/analytics        /api/group-orders      │
│  ┌───────────────┐    ┌─────────────────┐   ┌──────────────────┐   │
│  │ VendorRouter  │    │ AnalyticsRouter │   │ GroupOrderRouter │   │
│  │ • POST /reg.  │    │ • GET /restock  │   │ • POST /submit   │   │
│  │ • GET /nearby │    │ • POST /model-  │   │ • GET  /active   │   │
│  │ • GET /all    │    │       update    │   └────────┬─────────┘   │
│  └──────┬────────┘    │ • GET /items    │            │             │
│         │             │ • GET /gnn      │            │             │
│         │             └────────┬────────┘            │             │
│         │                      │                      │             │
│         │             ┌────────┴────────┐             │             │
│         │             │  AI  Analytics  │             │             │
│         │             │ ┌─────────────┐ │             │             │
│         │             │ │QLearningRL  │ │             │             │
│         │             │ │LSTMAnalyzer │ │             │             │
│         │             │ │GNNAnalyzer  │ │             │             │
│         │             │ └─────────────┘ │             │             │
│         │             └─────────────────┘             │             │
│  /api/chat                                            │             │
│  ┌───────────────┐                                    │             │
│  │  ChatRouter   │──► Google Gemini 2.0 Flash API     │             │
│  │ • POST /msg   │    (httpx async)                   │             │
│  └───────────────┘                                    │             │
└──────────┬────────────────────────────────────────────┘             │
           │  Motor (async)                             │             │
           ▼                                            ▼             │
┌─────────────────────────────────────────────────────────────────────┐
│                  MongoDB Atlas  (vendorgroove DB)                   │
│                                                                     │
│   vendors collection            group_orders collection             │
│   ┌──────────────────────┐      ┌──────────────────────────────┐   │
│   │ _id          ObjectId│      │ _id               ObjectId   │   │
│   │ shop_name    string  │      │ vendor_name       string     │   │
│   │ location     GeoJSON │      │ produce_type      string     │   │
│   │   type: "Point"      │      │ quantity          float      │   │
│   │   coordinates:[lng,  │      │ unit              string     │   │
│   │              lat]    │      │ delivery_window   string     │   │
│   │ created_at   Date    │      │ location          GeoJSON    │   │
│   │ is_active    bool    │      │ status            string     │   │
│   │                      │      │ created_at / expires_at Date │   │
│   │ INDEX: 2dsphere      │      │ INDEX: 2dsphere              │   │
│   └──────────────────────┘      └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data-flow highlights

| Flow | Path |
|---|---|
| Vendor onboarding | Browser → `POST /api/vendors/register` → MongoDB insert → Leaflet map pin |
| Restock advice | Browser → `GET /api/analytics/restock` → QL agent Q-table lookup → JSON |
| Active-learning update | Browser → `POST /api/analytics/model-update` → ε-greedy Bellman update |
| LSTM pricing | Browser → `GET /api/analytics/items` → LSTM anomaly score → badge + markdown |
| GNN insights | Browser → `GET /api/analytics/gnn` → GCN forward pass → trending / identity / trials |
| Group buying | Browser → `POST /api/group-orders/submit` → MongoDB `$near` + Haversine → truck manifest |
| AI chat | Browser → `POST /api/chat/message` → Gemini 2.0 Flash → Midori reply |

---

## Setup

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.11 + |
| pip / venv | bundled with Python |
| MongoDB Atlas account | free tier works |
| Google AI Studio key | for Gemini chat |

### 1 — Clone & create virtualenv

```bash
git clone <repo-url>
cd "Google hackthon Selling Agent"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment

Create `.env` in the project root:

```dotenv
MONGODB_URI=mongodb+srv://<user>:<password>@cluster0.xxxx.mongodb.net/vendorgroove?retryWrites=true&w=majority
APP_HOST=0.0.0.0
APP_PORT=1234
GEMINI_API_KEY=<your-google-ai-studio-key>
```

> If MongoDB is unavailable the app boots in **analytics-only mode** — all AI analytics endpoints still work; vendor registration and group-order matching are disabled.

---

## Run

```bash
# Development (hot-reload)
python -m uvicorn app.main:app --host 0.0.0.0 --port 1234 --reload

# Or via the __main__ entry-point
python -m app.main
```

Verify the server is healthy:

```bash
curl http://localhost:1234/health
# → {"status":"ok","service":"VendorGroove Net"}
```

Open the app in your browser:

| URL | Page |
|---|---|
| `http://localhost:1234/` | Onboarding (enter shop name, allow location) |
| `http://localhost:1234/dashboard` | Main dashboard |
| `http://localhost:1234/docs` | Auto-generated Swagger UI |
| `http://localhost:1234/redoc` | ReDoc API reference |

---

## Demo Flow

### Step 1 — Onboarding (`/`)

1. Enter your **shop name** (e.g. "Fresh Valley Stand").
2. Click **Allow** when the browser requests your location (falls back to NYC if denied).
3. A Leaflet map renders with a pin for your stand and any nearby vendors.
4. A sync animation plays, then the app redirects to `/dashboard`.

### Step 2 — Home tab (Midori AI)

The **Live2D VTuber avatar** (Midori) greets you.  
Type a question — e.g. *"How should I price my strawberries today?"* — and Midori replies via Gemini 2.0 Flash with actionable sales advice.

### Step 3 — Smart Restock tab

| Panel | What it shows |
|---|---|
| **Today's Status** | Current inventory state, sales velocity, weather signal |
| **Tomorrow's Recommendation** | RL recommended order quantity + confidence % |
| **RL Model Control** | Hit "Update Model" to feed in actual sales and watch TD-error, Q-value, and ε update live |

### Step 4 — Item Management tab

Six produce cards (Strawberries, Tomatoes, Spinach, Blueberries, Carrots, Zucchini) each show:

- **LSTM anomaly badge**: `STEADY` / `⚡ Flash Sale −20%` / `🚫 HALT SALES`
- Current vs base price
- 14-day sales sparkline + tomorrow's forecast
- LSTM gate internals (forget / input / output / cell)

Toggle **GNN Neighbourhood Insights** to see trending SKUs across the local vendor graph, your competitive identity, and low-risk trial items.

### Step 5 — Group Buying tab

1. Fill in: produce type, quantity, unit (lbs / bags / boxes / crates), delivery window.
2. Click **Find Match** — the backend queries MongoDB `$near` (8 047 m radius) for vendors sharing the same delivery window.
3. A **truck manifest** card lists all matched vendors and totals.
4. A **Leaflet map** pins all participants.
5. Click **Dispatch** to lock the order.

---

## Repo Layout

```
Google hackthon Selling Agent/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, lifespan, router registration
│   ├── database.py              # Motor AsyncIOMotorClient + db handle
│   │
│   ├── models/
│   │   └── vendor.py            # Pydantic: GeoLocation, VendorCreate, Vendor
│   │
│   ├── routes/
│   │   ├── vendors.py           # /api/vendors  (register, nearby, all)
│   │   ├── analytics.py         # /api/analytics (restock, model-update, items, gnn)
│   │   ├── group_orders.py      # /api/group-orders (submit, active)
│   │   └── chat.py              # /api/chat  (Gemini message)
│   │
│   ├── analytics/
│   │   ├── rl_model.py          # QLearningRestockAgent (Q-table, ε-greedy, Bellman)
│   │   ├── lstm_model.py        # LSTMTimeSeriesAnalyzer (anomaly + dynamic pricing)
│   │   └── gnn_model.py         # GNNNeighborhoodAnalyzer (GCN, attention, trending)
│   │
│   ├── templates/
│   │   ├── index.html           # Onboarding page (Leaflet, geoloc, vendor reg)
│   │   └── dashboard.html       # Main dashboard (Chart.js, Live2D, Leaflet, Tailwind)
│   │
│   └── static/
│       └── avatar/              # Live2D model assets (Midori 小恶魔)
│           ├── 小恶魔.moc3
│           ├── 小恶魔.model3.json
│           ├── 小恶魔.physics3.json
│           ├── 1-7.exp3.json    # Expression files
│           └── textures/
│               └── texture_00.png
│
├── .env                         # MONGODB_URI, GEMINI_API_KEY (not committed)
├── requirements.txt
└── README.md
```

---

## API Surface

All endpoints return `application/json`. Base URL: `http://localhost:1234`.

---

### Vendors  `/api/vendors`

#### `POST /api/vendors/register`
Register a new vendor with a geospatial location.

**Request body**
```json
{
  "shop_name": "Fresh Valley Stand",
  "location": {
    "type": "Point",
    "coordinates": [-74.0060, 40.7128]
  }
}
```

**Response `200`**
```json
{
  "id": "507f1f77bcf86cd799439011",
  "message": "Vendor registered"
}
```

---

#### `GET /api/vendors/nearby`
Find up to 20 vendors within a radius (MongoDB `$near`).

| Query param | Type | Default | Description |
|---|---|---|---|
| `lat` | float | required | Latitude |
| `lng` | float | required | Longitude |
| `radius` | int | 5000 | Search radius in metres |

**Response `200`** — array of vendor objects
```json
[
  {
    "_id": "507f1f77bcf86cd799439011",
    "shop_name": "Fresh Valley Stand",
    "location": {"type": "Point", "coordinates": [-74.006, 40.7128]},
    "created_at": "2026-06-11T10:00:00",
    "is_active": true
  }
]
```

---

#### `GET /api/vendors/all`
Return up to 50 registered vendors.

**Response `200`** — same shape as `/nearby`

---

### Analytics  `/api/analytics`

#### `GET /api/analytics/restock`
Q-Learning agent recommendation for tomorrow's restock quantity.

**Response `200`**
```json
{
  "state": {
    "inventory_change": -5,
    "sales_velocity": 55,
    "price_delta": 0.1
  },
  "prediction": {
    "recommended_quantity": 100,
    "state_key": "normal_inv_normal_sales_cloudy",
    "policy": "exploit",
    "q_value": 8.5,
    "confidence": 94,
    "training_episodes": 1247,
    "model_accuracy": 87.3,
    "epsilon": 0.1,
    "last_update": "2026-06-11 10:00:00"
  }
}
```

---

#### `POST /api/analytics/model-update`
Trigger an active-learning Bellman update with real sales data.

**Request body**
```json
{ "actual_sales": 92 }
```

**Response `200`**
```json
{
  "reward": 7.2,
  "td_error": 0.38,
  "delta_q": 0.19,
  "new_accuracy": 88.1,
  "new_epsilon": 0.095,
  "episodes": 1248,
  "state_key": "normal_inv_normal_sales_cloudy"
}
```

---

#### `GET /api/analytics/items`
LSTM anomaly detection and dynamic pricing for all tracked SKUs.

**Response `200`**
```json
{
  "items": [
    {
      "id": "strawberries",
      "name": "Strawberries",
      "emoji": "🍓",
      "base_price": 4.99,
      "unit": "pint",
      "current_price": 3.99,
      "status": "FLASH_SALE",
      "badge": "⚡ Flash Sale −20%",
      "anomaly_score": 0.82,
      "decay_rate": 0.12,
      "sales_history": [45, 52, 48, 41, 38, 35, 30, 28, 25, 22, 20, 18, 15, 12],
      "forecast_tomorrow": 38.5,
      "lstm_gates": {
        "forget": 0.73,
        "input": 0.81,
        "output": 0.69,
        "cell": 0.77
      },
      "tip": "Move inventory fast — demand dropping"
    }
  ]
}
```

**Anomaly thresholds**

| Score | Status | Action |
|---|---|---|
| 0 – 0.55 | `STEADY` | Normal pricing |
| 0.55 – 0.82 | `FLASH_SALE` | 15–25 % markdown |
| > 0.82 | `HALT_SALES` | Recommend alternatives |

---

#### `GET /api/analytics/gnn`
GNN neighbourhood graph analysis — trending items, competitive identity, and trial SKU candidates.

**Response `200`**
```json
{
  "graph": {
    "nodes": 6,
    "edges": 15,
    "strong_edges": 8,
    "avg_edge_weight": 0.52,
    "embed_norm": 1.24,
    "layers": 2,
    "attention_heads": 4
  },
  "trending": [
    {
      "name": "Dragon Fruit",
      "emoji": "🐉",
      "velocity": "+34%",
      "competitors": 1,
      "margin": "High"
    }
  ],
  "identity": "Focus on organic leafy greens — 0 competitors within 0.3 mi",
  "trials": [
    {
      "item": "Japanese Sweet Potato",
      "emoji": "🍠",
      "risk": "Low",
      "demand": "High"
    }
  ],
  "network_health": 87.3
}
```

---

### Group Orders  `/api/group-orders`

#### `POST /api/group-orders/submit`
Submit an order and auto-match with nearby vendors sharing the same delivery window.

**Request body**
```json
{
  "vendor_name": "Fresh Valley Stand",
  "produce_type": "Tomatoes",
  "quantity": 50,
  "unit": "lbs",
  "delivery_window": "06:00 AM – 09:00 AM",
  "location": {
    "type": "Point",
    "coordinates": [-74.0060, 40.7128]
  }
}
```

**Response `200`**
```json
{
  "success": true,
  "order_id": "507f1f77bcf86cd799439011",
  "truck_id": "A9",
  "your_order": {
    "produce_type": "Tomatoes",
    "quantity": 50,
    "unit": "lbs"
  },
  "matched_vendors": [
    {
      "vendor_name": "Green Leaf Market",
      "lat": 40.7156,
      "lng": -74.0038,
      "distance_mi": 0.25
    }
  ],
  "total_in_truck": 3,
  "service_radius_mi": 5,
  "pickup_eta": "Tomorrow 5:30 AM",
  "algorithm": "$near · maxDistance=8047m · same delivery_window"
}
```

---

#### `GET /api/group-orders/active`
Return up to 50 pending group orders.

**Response `200`** — array of order objects
```json
[
  {
    "_id": "507f1f77bcf86cd799439011",
    "vendor_name": "Fresh Valley Stand",
    "produce_type": "Tomatoes",
    "quantity": 50,
    "unit": "lbs",
    "delivery_window": "06:00 AM – 09:00 AM",
    "location": {"type": "Point", "coordinates": [-74.006, 40.7128]},
    "status": "pending",
    "created_at": "2026-06-11T10:00:00",
    "expires_at": "2026-06-12T10:00:00"
  }
]
```

---

### Chat  `/api/chat`

#### `POST /api/chat/message`
Send a message to Midori, the Gemini-powered AI sales assistant.

**Request body**
```json
{
  "message": "How should I price my strawberries today?",
  "shop": "Fresh Valley Stand",
  "inventory": "40 pints of strawberries, last sold at $4.99"
}
```

**Response `200`**
```json
{
  "reply": "Hey! With 40 pints on hand and demand softening, I'd drop to $3.99 and bundle two pints for $7 — that move-fast strategy usually clears 60 % of stock before noon. Good luck! 🍓"
}
```

---

### Root endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Onboarding HTML page |
| `GET` | `/dashboard` | Dashboard HTML page |
| `GET` | `/health` | `{"status":"ok","service":"VendorGroove Net"}` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI 0.115 + Uvicorn |
| Database | MongoDB Atlas (Motor async driver) |
| AI — restock | Q-Learning (custom, no external ML lib) |
| AI — pricing | 2-layer LSTM (custom NumPy implementation) |
| AI — neighbourhood | 2-layer GCN with 4-head attention (custom) |
| AI — chat | Google Gemini 2.0 Flash (via httpx) |
| Frontend framework | Vanilla JS + TailwindCSS |
| Charts | Chart.js |
| Maps | Leaflet.js |
| Avatar | Live2D Cubism SDK + PixiJS |
| Validation | Pydantic v2 |
| Config | python-dotenv |
