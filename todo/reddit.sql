-- ─────────────────────────────────────────────────────────────────────────────
-- migrations/reddit.sql
-- Add to your existing migration runner in core/models.py
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS reddit_posts (
    id              SERIAL PRIMARY KEY,
    post_id         TEXT        NOT NULL UNIQUE,
    subreddit       TEXT        NOT NULL,
    author          TEXT,
    title           TEXT,
    body            TEXT,
    url             TEXT,
    score           INTEGER     DEFAULT 0,
    num_comments    INTEGER     DEFAULT 0,
    is_comment      BOOLEAN     DEFAULT FALSE,
    asset           TEXT        NOT NULL,       -- GC | NQ | ES | 6E | BTC | ETH | SOL
    sentiment_score FLOAT,                      -- VADER compound  (-1.0 → +1.0)
    sentiment_label TEXT,                       -- bullish | bearish | neutral
    upvote_ratio    FLOAT,
    created_utc     TIMESTAMPTZ NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reddit_posts_asset      ON reddit_posts (asset);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_created    ON reddit_posts (created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_asset_time ON reddit_posts (asset, created_utc DESC);


-- Useful queries for backtesting / analysis
-- ─────────────────────────────────────────────────────────────────────────────

-- Hourly sentiment distribution per asset (last 7 days)
SELECT
    asset,
    date_trunc('hour', created_utc) AS hour,
    COUNT(*)                         AS mentions,
    AVG(sentiment_score)             AS avg_sentiment,
    SUM(CASE WHEN sentiment_label = 'bullish' THEN 1 ELSE 0 END)::float / COUNT(*) AS bull_ratio
FROM reddit_posts
WHERE created_utc > NOW() - INTERVAL '7 days'
GROUP BY asset, hour
ORDER BY asset, hour DESC;

-- Mention spikes vs 24-hr baseline (identifies unusual social activity)
WITH baseline AS (
    SELECT asset, COUNT(*) / 24.0 AS hourly_avg
    FROM reddit_posts
    WHERE created_utc > NOW() - INTERVAL '24 hours'
    GROUP BY asset
),
recent AS (
    SELECT asset, COUNT(*) AS last_15m
    FROM reddit_posts
    WHERE created_utc > NOW() - INTERVAL '15 minutes'
    GROUP BY asset
)
SELECT r.asset,
       r.last_15m,
       b.hourly_avg / 4 AS expected_15m,
       r.last_15m / NULLIF(b.hourly_avg / 4, 0) AS velocity_ratio
FROM recent r
JOIN baseline b USING (asset)
ORDER BY velocity_ratio DESC;

-- Sentiment vs price movement correlation helper (join with your bars table)
-- SELECT rp.asset, rp.hour, rp.avg_sentiment, b.close - b.open AS bar_move
-- FROM (hourly sentiment CTE above) rp
-- JOIN bars b ON b.symbol = rp.asset AND b.ts = rp.hour
-- ORDER BY rp.asset, rp.hour;
