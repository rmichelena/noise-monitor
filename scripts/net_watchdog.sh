#!/bin/sh
# Watchdog de conectividad: si el host pierde gateway o internet durante
# varios chequeos, reconecta Wi-Fi (networkd + suplicante netplan).
# Ejecutado por noise-net-watchdog.timer cada 60 s.

set -u

GW="${GW:-192.168.1.1}"
PROBE="${PROBE:-1.1.1.1}"
STATE_DIR=/run/noise-net-watchdog
mkdir -p "$STATE_DIR"
FAILS_FILE="$STATE_DIR/failures"
LAST_FILE="$STATE_DIR/last_reconnect"

fails=$(cat "$FAILS_FILE" 2>/dev/null || echo 0)

if ping -c2 -W2 "$GW" >/dev/null 2>&1 && ping -c1 -W2 "$PROBE" >/dev/null 2>&1; then
    if [ "$fails" -gt 0 ]; then
        logger -t noise-net-watchdog "conectividad restaurada tras $fails fallo(s)"
    fi
    echo 0 > "$FAILS_FILE"
    # Muestra periódica de señal para ver tendencia en el journal.
    n=$(cat "$STATE_DIR/count" 2>/dev/null || echo 0)
    n=$((n + 1))
    echo "$n" > "$STATE_DIR/count"
    if [ $((n % 60)) -eq 0 ]; then
        sig=$(iw dev wlp2s0 link 2>/dev/null | awk '/signal:/{print $2, $3}')
        logger -t noise-net-watchdog "señal wifi: ${sig:-desconocida}"
    fi
    exit 0
fi

fails=$((fails + 1))
echo "$fails" > "$FAILS_FILE"
logger -t noise-net-watchdog "sin conectividad (fallo $fails, gw=$GW probe=$PROBE)"

if [ "$fails" -lt 3 ]; then
    exit 0
fi

now=$(date +%s)
last=$(cat "$LAST_FILE" 2>/dev/null || echo 0)
if [ $((now - last)) -lt 300 ]; then
    logger -t noise-net-watchdog "reconexión omitida: última hace $((now - last)) s"
    exit 0
fi
echo "$now" > "$LAST_FILE"

logger -t noise-net-watchdog "reconectando red (systemd-networkd + netplan-wpa)"
systemctl restart systemd-networkd
sleep 3
systemctl restart netplan-wpa-wlp2s0.service
sleep 10
if ping -c2 -W2 "$GW" >/dev/null 2>&1; then
    logger -t noise-net-watchdog "gateway recuperado tras reconectar"
    echo 0 > "$FAILS_FILE"
else
    logger -t noise-net-watchdog "gateway sigue sin responder tras reconectar"
fi
