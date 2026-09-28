#!/usr/bin/env python3
"""Servidor stdlib para puntuar respuestas de Kateto y exportar pares ORPO.

Uso: python3 server.py [puerto]   (default 8765)
Archivos (mismo dir): candidatos.jsonl, preferencias.jsonl, pares_orpo.jsonl
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

BASE = os.path.dirname(os.path.abspath(__file__))
CANDS = os.path.join(BASE, "candidatos.jsonl")
PREFS = os.path.join(BASE, "preferencias.jsonl")
PARES = os.path.join(BASE, "pares_orpo.jsonl")
INDEX = os.path.join(BASE, "index.html")


def leer_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def estado():
    cands = leer_jsonl(CANDS)
    votos = leer_jsonl(PREFS)
    votados = {v["id"] for v in votos if "id" in v}
    empates = sum(1 for v in votos if v.get("empate"))
    pares = leer_jsonl(PARES)
    return {
        "total_candidatos": len(cands),
        "votos": len(votos),
        "empates": empates,
        "pares_exportados": len(pares),
        "pendientes": len([c for c in cands if c.get("id") not in votados]),
        "ultimo_id": votos[-1]["id"] if votos else None,
    }


def siguiente():
    cands = leer_jsonl(CANDS)
    votos = leer_jsonl(PREFS)
    votados = {v["id"] for v in votos if "id" in v}
    for c in cands:
        if c.get("id") not in votados:
            return c
    return None


def exportar():
    votos = leer_jsonl(PREFS)
    pares = []
    for v in votos:
        if v.get("empate"):
            continue
        r = v["respuestas"]
        p = v["puntajes"]
        m = v["mejor"]
        if m is None or not (0 <= m < len(r)):
            continue
        for j in range(len(r)):
            if j != m and p[m] > p[j]:
                pares.append({"id": f'{v["id"]}:{m}>{j}',
                              "prompt": v["prompt"],
                              "chosen": r[m], "rejected": r[j]})
    with open(PARES, "w", encoding="utf-8") as f:
        for par in pares:
            f.write(json.dumps(par, ensure_ascii=False) + "\n")
    return {"pares": len(pares), "prompts": len({p["prompt"] for p in pares})}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return None

    def do_GET(self):
        u = urlparse(self.path).path
        if u in ("/", "/index.html"):
            try:
                with open(INDEX, "rb") as f:
                    b = f.read()
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif u == "/api/data":
            c = siguiente()
            e = estado()
            self._json({"actual": c, "pendientes": e["pendientes"],
                        "votos": e["votos"], "total": e["total_candidatos"]})
        elif u == "/api/estado":
            self._json(estado())
        elif u == "/api/export":
            self._json({"ok": True, **exportar(), **estado()})
        else:
            self._json({"error": "no existe"}, 404)

    def do_POST(self):
        u = urlparse(self.path).path
        if u == "/api/votar":
            data = self._body()
            if data is None:
                return self._json({"error": "JSON invalido"}, 400)
            cid, puntajes, mejor, empate = (data.get("id"), data.get("puntajes"),
                                            data.get("mejor"), bool(data.get("empate")))
            cands = {c["id"]: c for c in leer_jsonl(CANDS)}
            c = cands.get(cid)
            if c is None:
                return self._json({"error": "id desconocido"}, 400)
            n = len(c["respuestas"])
            if (not isinstance(puntajes, list) or len(puntajes) != n
                    or any(p not in (1, 2, 3, 4, 5) for p in puntajes)):
                return self._json({"error": "puntajes: uno por respuesta, 1-5"}, 400)
            if empate:
                if len(set(puntajes)) != 1:
                    return self._json({"error": "empate exige mismo puntaje"}, 400)
                mejor = None
            else:
                if not isinstance(mejor, int) or not (0 <= mejor < n):
                    return self._json({"error": "elegi la mejor"}, 400)
                if not all(puntajes[mejor] > p for j, p in enumerate(puntajes) if j != mejor):
                    return self._json({"error": "la mejor debe superar a las demas"}, 400)
            append_jsonl(PREFS, {"ts": int(time.time()), "id": cid, "prompt": c["prompt"],
                                 "respuestas": c["respuestas"], "puntajes": puntajes,
                                 "mejor": mejor, "empate": empate})
            nxt = siguiente()
            return self._json({"ok": True, "siguiente": nxt, **estado()})
        if u == "/api/deshacer":
            votos = leer_jsonl(PREFS)
            if not votos:
                return self._json({"error": "nada que deshacer"}, 400)
            quitado = votos.pop()
            with open(PREFS, "w", encoding="utf-8") as f:
                for v in votos:
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
            return self._json({"ok": True, "quitado": quitado["id"], **estado()})
        if u == "/api/export":
            return self._json({"ok": True, **exportar(), **estado()})
        return self._json({"error": "no existe"}, 404)


if __name__ == "__main__":
    puerto = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", puerto), H)
    print(f"kateto-rlhf en http://127.0.0.1:{puerto}", flush=True)
    srv.serve_forever()
