## Inspiration
Independent vegetable vendors rely on intuition for every decision — how much to order, when to discount, whether a truck is worth the cost alone. We wanted to give small stands the same AI-powered tools that big grocery chains take for granted.

## What it does
VendorGroove Net is a web dashboard with five features: a Q-Learning agent that recommends tomorrow's restock quantity; an LSTM anomaly detector that triggers automatic price markdowns; a Graph Neural Network that surfaces trending SKUs and competitive niches; a group-buying coordinator that pools nearby vendors into shared truck manifests; and Midori, a Gemini 2.0 Flash AI assistant embodied as a Live2D avatar.

## How we built it
FastAPI + Uvicorn backend, MongoDB Atlas for geospatial vendor and order storage, three custom ML models written in pure Python (no PyTorch or TensorFlow), Gemini 2.0 Flash for chat, and a vanilla JS frontend using TailwindCSS, Chart.js, Leaflet.js, and PixiJS + Live2D.

## Challenges we ran into
Keeping ML model state across stateless HTTP requests, integrating Live2D without a build pipeline, making raw model outputs (Q-values, anomaly scores) legible to non-technical vendors, and producing meaningful group-buy manifests with sparse real-world data.

## Accomplishments that we're proud of
Three production-grade AI models with zero external ML dependencies, a fully connected end-to-end stack with no placeholder screens, and real geospatial queries against live MongoDB records.

## What we learned
Tabular Q-Learning is practical without a training cluster. Prompt engineering determines AI product quality more than model choice. Coordination problems (shared trucks) create more value than optimization problems (better forecasts).

## What's next for Vendors Selling Agent
Persistent model state across sessions, JWT authentication, real LSTM training on USDA market data, PWA offline support, direct supplier API ordering, and cooperative network analytics across the full vendor graph.
