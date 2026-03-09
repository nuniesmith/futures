# ─────────────────────────────────────────────────────────────────────────────
# WIRING GUIDE
# How to plug reddit_watcher into your existing codebase
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. .env.example additions ─────────────────────────────────────────────────
#
# Get a free read-only Reddit app at https://www.reddit.com/prefs/apps
# Create an app → type = "script" → redirect URI = http://localhost
#
# REDDIT_CLIENT_ID=your_client_id_here
# REDDIT_CLIENT_SECRET=your_client_secret_here
# REDDIT_POLL_INTERVAL=120          # seconds between polls (default 120)
# USE_FINBERT=false                 # set true to use ProsusAI/finbert (GPU recommended)


# ── 2. pyproject.toml — add to [project].dependencies ────────────────────────
#
# "praw>=7.7",
# "vaderSentiment>=3.3",
# # optional FinBERT path:
# # "transformers>=4.40",


# ── 3. core/models.py  — add migration call ───────────────────────────────────
#
# In your existing run_migrations() or startup handler, add:
#
#     await conn.execute(open("migrations/reddit.sql").read())
#
# Or paste the reddit.sql DDL directly into your existing migration block.


# ── 4. scheduler.py  — add watcher task + aggregation job ────────────────────
#
# In your lifespan startup (or wherever you launch background tasks):

"""
from lib.integrations.reddit_watcher import RedditWatcher
from lib.analysis.reddit_sentiment import get_full_snapshot

# -- Start the scraper (runs forever, writes to Redis + Postgres)
watcher = RedditWatcher(
    redis=app.state.redis,
    pg_pool=app.state.pg_pool,
    mode="poll",   # switch to "stream" for lower latency on busy subs
)
asyncio.create_task(watcher.run())

# -- Scheduled aggregation (every 5 minutes, same cadence as your engine)
async def _reddit_aggregation_job():
    while True:
        try:
            await get_full_snapshot(app.state.redis)
        except Exception as exc:
            logger.warning("Reddit aggregation error: %s", exc)
        await asyncio.sleep(300)   # 5 minutes

asyncio.create_task(_reddit_aggregation_job())
"""


# ── 5. main app  — register the router ───────────────────────────────────────
#
# In your FastAPI app setup (same place you include analysis.py, journal.py etc):

"""
from lib.services.data.api.reddit import router as reddit_router
app.include_router(reddit_router)
"""


# ── 6. HTMX dashboard — add the panel ────────────────────────────────────────
#
# In your dashboard.py Jinja template, add wherever you want the panel:
#
#   <!-- Reddit Sentiment Panel — auto-refreshes every 2 minutes -->
#   <div hx-get="/htmx/reddit/panel"
#        hx-trigger="load, every 120s"
#        hx-swap="outerHTML">
#     <div style="color:#555">Loading Reddit sentiment…</div>
#   </div>
#
# For a single-asset deep-dive (e.g. on your NQ card):
#
#   <div hx-get="/htmx/reddit/asset/NQ"
#        hx-trigger="load, every 120s"
#        hx-swap="outerHTML">
#   </div>


# ── 7. SSE push (optional) ────────────────────────────────────────────────────
#
# In sse.py, alongside your existing signal push, add:
#
#   from lib.analysis.reddit_sentiment import get_asset_signal
#
#   for asset in ["GC", "NQ", "ES", "6E", "BTC", "ETH", "SOL"]:
#       sig = await get_asset_signal(asset, redis)
#       yield f"event: reddit_signal\ndata: {json.dumps({'asset': asset, **sig})}\n\n"
