##
## Custom Redis Dockerfile for Production
## Futures Trader - Cache & Pub/Sub Service
##
## Features:
## - Based on official redis:7-alpine for minimal footprint
## - Custom entrypoint that disables Transparent Huge Pages (THP) to avoid
##   latency spikes. THP is a /sys knob and cannot be set via Docker sysctls.
## - Kernel sysctls (vm.overcommit_memory, net.core.somaxconn) are applied
##   via the `sysctls:` directive in docker-compose.yml — no privileged mode needed.
## - Graceful fallback if THP cannot be disabled (logs a warning)
## - Delegates to official docker-entrypoint.sh for user switching & signal handling
##
## NOTE: The container no longer requires `privileged: true`. Sysctls are set
##       declaratively in docker-compose.yml. Only THP disable needs the
##       entrypoint (and will log a cosmetic warning if it cannot be applied).
##
## Build:
##   docker build -t nuniesmith/nuniesmith/fks:redis \
##     -f infrastructure/docker/services/redis/Dockerfile .
##

# =============================================================================
# STAGE 1: Builder - Prepare entrypoint script
# =============================================================================
FROM alpine:latest AS builder

LABEL stage="builder"

WORKDIR /build

COPY docker/redis/entrypoint.sh /build/entrypoint.sh

# Validate and prepare entrypoint
RUN set -ex && \
    # Ensure LF line endings (not CRLF)
    sed -i 's/\r$//' /build/entrypoint.sh && \
    chmod +x /build/entrypoint.sh && \
    # Sanity check — must start with shebang
    head -1 /build/entrypoint.sh | grep -q '^#!/' && \
    echo "Entrypoint script prepared successfully"

# =============================================================================
# STAGE 2: Production - Final Redis image
# =============================================================================
FROM redis:7.4-alpine AS production

LABEL maintainer="Futures Trader"
LABEL description="Redis cache & pub/sub for Futures Trader with kernel tuning for reliable persistence"
LABEL version="1.0.0"

# Copy our custom entrypoint (runs BEFORE the official docker-entrypoint.sh)
COPY --from=builder /build/entrypoint.sh /usr/local/bin/fks-entrypoint.sh

# Ensure the data directory exists with correct ownership
RUN mkdir -p /data && chown redis:redis /data

# Redis default port
EXPOSE 6379

# Health check — uses redis-cli ping (password injected via env at runtime)
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD redis-cli -a "${REDIS_PASSWORD:-}" --no-auth-warning ping | grep -q PONG || exit 1

# Our entrypoint handles kernel tuning, then delegates to the official entrypoint
ENTRYPOINT ["fks-entrypoint.sh"]

# Default command — can be overridden entirely via docker-compose `command:`
CMD ["redis-server"]
