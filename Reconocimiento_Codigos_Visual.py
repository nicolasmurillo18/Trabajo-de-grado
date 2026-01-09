import cv2
import pytesseract
import re
import pyodbc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()  # Carga el .env

# =====================================================
# CONFIG
# =====================================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
REGEX_ID = re.compile(r"\b(\d{5,8})\b")

OCR_NUM_CONFIG = (
    "--oem 3 "
    "--dpi 300 "                          # <-- clave
    "-c user_defined_dpi=300 "            # <-- clave
    "-c tessedit_char_whitelist=0123456789 "
    "-c classify_bln_numeric_mode=1 "
    "-c load_system_dawg=0 -c load_freq_dawg=0 "
)


# PSMs típicos para números en etiqueta / línea / bloque
OCR_PSMS = (7, 8, 6, 13)

# =====================================================
# DEBUG VIEW
# =====================================================


def mostrar_roi(titulo: str, roi, ancho_max=1100):
    if roi is None or roi.size == 0:
        print(f"[DEBUG] {titulo}: ROI vacía.")
        return

    if len(roi.shape) == 2:
        roi_vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        roi_vis = roi.copy()

    h, w = roi_vis.shape[:2]
    if w > ancho_max:
        s = ancho_max / float(w)
        roi_vis = cv2.resize(roi_vis, (int(w*s), int(h*s)))

    cv2.imshow(titulo, roi_vis)
    cv2.waitKey(0)
    cv2.destroyWindow(titulo)


def mostrar_pack_binarizaciones(nombre: str, gray_roi):
    """Útil cuando quieras ver por qué falla el OCR."""
    ths = binarizaciones(gray_roi)
    mostrar_roi(f"{nombre} | GRAY", gray_roi)
    for i, th in enumerate(ths, 1):
        mostrar_roi(f"{nombre} | BIN_{i}", th)


# =====================================================
# DB VALIDATION (cached)
# =====================================================
def existe_id_en_bd(conn, strict: bool = True):
    """
    strict=True:
      - Si conn es None (o falla la consulta), retorna False.
      - Es decir: NUNCA valida si no hay BD.
    strict=False:
      - Si conn es None, retorna True (modo antiguo: útil solo para pruebas sin BD).
    """
    cache = {}

    def _existe(pid: str) -> bool:
        if conn is None:
            return False if strict else True

        if pid in cache:
            return cache[pid]

        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM rob.Producto WHERE Id_producto = ?;", (int(pid),))
            ok = cur.fetchone() is not None
        except Exception as e:
            # En modo estricto, ante error de BD, no validamos
            ok = False if strict else True

        cache[pid] = ok
        return ok

    return _existe


# =====================================================
# HEURISTICS (trash filter)
# =====================================================
def es_candidato_basura(pid: str) -> bool:
    # EAN-13 completo (por si se cuela)
    if len(pid) == 13 and pid.startswith(("770", "777", "778", "779")):
        return True

    # Fechas comunes comprimidas a 8 dígitos:
    # ddmmyyyy (e.g. 19052025, 12102024)
    if len(pid) == 8:
        dd = int(pid[0:2])
        mm = int(pid[2:4])
        yyyy = int(pid[4:8])
        if 1 <= dd <= 31 and 1 <= mm <= 12 and 2000 <= yyyy <= 2099:
            return True

        # yyyymmdd (e.g. 20250519)
        yyyy = int(pid[0:4])
        mm = int(pid[4:6])
        dd = int(pid[6:8])
        if 2000 <= yyyy <= 2099 and 1 <= mm <= 12 and 1 <= dd <= 31:
            return True

    return False


# =====================================================
# PREPROCESS
# =====================================================
def preparar_gray(imagen_bgr, target=1600):
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)

    h, w = gray.shape[:2]
    m = max(h, w)
    if m < target:
        s = min(2.0, max(1.0, target / float(m)))
        if s > 1.05:
            gray = cv2.resize(gray, None, fx=s, fy=s,
                              interpolation=cv2.INTER_CUBIC)

    return gray


def binarizaciones(gray_roi):
    if gray_roi is None or gray_roi.size == 0:
        return []

    out = []

    # --- (A) Otsu normal / invertido (igual que tienes)
    g = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(th)
    out.append(cv2.bitwise_not(th))

    # --- (B) Adaptive normal / invertido (igual que tienes)
    th2 = cv2.adaptiveThreshold(
        gray_roi, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )
    out.append(th2)
    out.append(cv2.bitwise_not(th2))

    # --- (C) Unsharp + Otsu (igual que tienes)
    blur = cv2.GaussianBlur(gray_roi, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray_roi, 1.6, blur, -0.6, 0)
    g2 = cv2.GaussianBlur(sharp, (3, 3), 0)
    _, th3 = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(th3)
    out.append(cv2.bitwise_not(th3))

    # =====================================================
    # NUEVO (D) Normalización + Otsu (mejora contraste global)
    # =====================================================
    norm = cv2.normalize(gray_roi, None, 0, 255, cv2.NORM_MINMAX)
    gn = cv2.GaussianBlur(norm, (3, 3), 0)
    _, th4 = cv2.threshold(gn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(th4)
    out.append(cv2.bitwise_not(th4))

    # =====================================================
    # NUEVO (E) Dilation suave sobre Otsu (repara dígitos partidos)
    # =====================================================
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th_d = cv2.dilate(th, k, iterations=1)
    out.append(th_d)
    out.append(cv2.bitwise_not(th_d))

    return out


def recorte_bordes(gray, pct=0.02):
    """Recorta un porcentaje pequeño de borde para evitar dígitos pegados al límite."""
    if gray is None or gray.size == 0:
        return gray
    h, w = gray.shape[:2]
    dx = max(1, int(pct * w))
    dy = max(1, int(pct * h))
    if w - 2*dx <= 5 or h - 2*dy <= 5:
        return gray
    return gray[dy:h-dy, dx:w-dx]


def morfologia_suave(th):
    """
    Cierre suave para unir trazos rotos (especialmente en dígitos extremos).
    Se aplica sobre binarizadas, no sobre gris.
    """
    if th is None or th.size == 0:
        return th
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)


# =====================================================
# OCR CORE (votación robusta)
# =====================================================
def _extraer_candidatos_desde_data(data: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Extrae candidatos 5-8 dígitos desde image_to_data:
    - tokens individuales
    - concatenaciones por línea (muy clave cuando OCR parte el número)
    Retorna lista (pid, conf_aprox).
    """
    items = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        d = re.sub(r"\D", "", txt)
        if not d:
            continue
        try:
            conf = float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
        except:
            conf = -1.0

        ln = data.get("line_num", [0] * n)[i]
        left = data.get("left", [0] * n)[i]
        items.append((ln, left, d, conf))

    if not items:
        return []

    # agrupa por línea
    items.sort(key=lambda x: (x[0], x[1]))
    por_linea: Dict[int, List[Tuple[int, str, float]]] = {}
    for ln, left, d, conf in items:
        por_linea.setdefault(ln, []).append((left, d, conf))

    cands: List[Tuple[str, float]] = []

    # 1) tokens 5-8
    for ln in por_linea:
        for left, d, conf in por_linea[ln]:
            if 5 <= len(d) <= 8:
                cands.append((d, conf))

    # 2) concatenaciones dentro de la misma línea
    for ln in por_linea:
        toks = por_linea[ln]
        m = len(toks)
        for i in range(m):
            acc = ""
            confs = []
            for j in range(i, m):
                acc += toks[j][1]
                confs.append(toks[j][2])
                if len(acc) > 8:
                    break
                if 5 <= len(acc) <= 8:
                    cands.append((acc, min(confs)))

    # dedup conservando mejor conf
    best: Dict[str, float] = {}
    for pid, conf in cands:
        if pid not in best or conf > best[pid]:
            best[pid] = conf

    out = [(pid, best[pid]) for pid in best.keys()]
    out.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
    return out


def preparar_para_ocr(gray):
    """
    Mejora local para OCR:
    - reescala un poco (mejora dígitos finales)
    - reduce ruido sin borrar bordes
    - sharpen leve
    """
    if gray is None or gray.size == 0:
        return gray

    h, w = gray.shape[:2]

    if h < 220:
        s = 220 / float(h)
        s = min(max(s, 1.0), 1.8)
        if s > 1.05:
            gray = cv2.resize(gray, None, fx=s, fy=s,
                              interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    # NUEVO: normalización final (suave)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    return gray


def ocr_votacion_id(gray_roi, existe_bd: Optional[Callable[[str], bool]] = None) -> Tuple[Optional[str], float, str]:
    """
    Retorna: (pid_ganador|None, conf_mejor, tag)
    - Hace votación en múltiples binarizaciones y PSMs
    - Revisa TOP (franja superior) y FULL
    - Si existe_bd: prioriza IDs válidos
    """
    if gray_roi is None or gray_roi.size == 0:
        return None, -1.0, "EMPTY_ROI"

    # NUEVO: recorte leve de bordes para estabilizar primer/último dígito
    gray_roi = recorte_bordes(gray_roi, pct=0.02)

    H = gray_roi.shape[0]
    roi_top = gray_roi[: int(0.45 * H), :]

    votos: Dict[str, Dict[str, Any]] = {}

    for base_img, region_tag in ((roi_top, "TOP"), (gray_roi, "FULL")):
        # NUEVO: también recortamos bordes de cada región
        base_img = recorte_bordes(base_img, pct=0.02)
        base_img = preparar_para_ocr(base_img)

        for th in binarizaciones(base_img):
            # NUEVO: cierre morfológico suave (repara trazos rotos)
            th = morfologia_suave(th)

            for psm in OCR_PSMS:
                data = pytesseract.image_to_data(
                    th,
                    config=f"{OCR_NUM_CONFIG} --psm {psm}",
                    output_type=pytesseract.Output.DICT
                )
                cands = _extraer_candidatos_desde_data(data)
                if not cands:
                    continue

                for pid, conf in cands[:3]:
                    if es_candidato_basura(pid):
                        continue
                    if pid not in votos:
                        votos[pid] = {"votes": 0, "best_conf": conf,
                                      "tag": f"{region_tag}/psm{psm}"}
                    votos[pid]["votes"] += 1
                    if conf > votos[pid]["best_conf"]:
                        votos[pid]["best_conf"] = conf
                        votos[pid]["tag"] = f"{region_tag}/psm{psm}"

    if not votos:
        return None, -1.0, "NO_MATCH"

    def score(pid: str):
        v = votos[pid]
        bd_ok = 1 if (existe_bd(pid) if existe_bd else True) else 0
        return (bd_ok, v["votes"], len(pid), v["best_conf"])

    ganador = max(votos.keys(), key=score)
    best_conf = float(votos[ganador]["best_conf"])
    best_votes = int(votos[ganador]["votes"])
    best_tag = str(votos[ganador]["tag"])

    # NUEVO: gate anti-alucinación cuando NO valida BD
    if existe_bd is not None:
        bd_ok = bool(existe_bd(ganador))
        if not bd_ok:
            # Ajusta estos umbrales si lo necesitas (son conservadores)
            if best_votes < 2 or best_conf < 35:
                return None, best_conf, f"LOW_EVIDENCE({best_tag})"

    return ganador, best_conf, best_tag

# =====================================================
# ROI CANDIDATES
# =====================================================


@dataclass
class RoiCandidate:
    name: str
    gray: any  # numpy array
    meta: str = ""


def encontrar_roi_etiqueta_blanca(imagen_bgr) -> Optional[RoiCandidate]:
    """
    Detector de etiqueta blanca (robusto y relativamente estricto).
    """
    H_img, W_img = imagen_bgr.shape[:2]

    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)

    mask_hsv = cv2.inRange(hsv, (0, 0, 160), (180, 170, 255))
    L, A, B = cv2.split(lab)
    mask_lab = cv2.inRange(L, 170, 255)
    mask = cv2.bitwise_and(mask_hsv, mask_lab)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
        iterations=2
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        area_rel = area / float(H_img * W_img)

        if area_rel < 0.012 or area_rel > 0.40:
            continue

        ratio = w / float(h + 1e-6)
        if ratio < 1.15 or ratio > 8.5:
            continue

        contour_area = cv2.contourArea(c)
        if contour_area <= 0:
            continue

        rectangularidad = contour_area / float(area + 1e-6)
        if rectangularidad < 0.72:
            continue

        hull = cv2.convexHull(c)
        solidez = contour_area / float(cv2.contourArea(hull) + 1e-6)
        if solidez < 0.88:
            continue

        # contenido interno: bordes (tinta/barcode) para evitar cartón claro
        roi = imagen_bgr[y:y+h, x:x+w]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g, 60, 160)
        edge_density = edges.mean() / 255.0
        if edge_density < 0.015:
            continue

        score = area * \
            (1.2 if (x > 0.25 * W_img and y < 0.75 * H_img) else 1.0)
        if best is None or score > best["score"]:
            best = {"x": x, "y": y, "w": w, "h": h,
                    "score": score, "edge": edge_density}

    if best is None:
        return None

    pad = 10
    x, y, w, h = best["x"], best["y"], best["w"], best["h"]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W_img, x + w + pad)
    y2 = min(H_img, y + h + pad)

    roi_bgr = imagen_bgr[y1:y2, x1:x2]
    roi_gray = preparar_gray(roi_bgr)
    return RoiCandidate(name="ETIQUETA_BLANCA", gray=roi_gray, meta=f"edge_density={best['edge']:.4f}")


def encontrar_rois_por_texto_oscuro(imagen_bgr, max_rois=3) -> List[RoiCandidate]:
    """
    Black-hat para proponer ROIs donde hay trazos oscuros (texto/números).
    """
    gray0 = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    blackhat = cv2.morphologyEx(gray0, cv2.MORPH_BLACKHAT, k)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    _, bw = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 11))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k2, iterations=2)

    contours, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray0.shape[:2]

    cand = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < 0.01 * (H * W):
            continue
        if w > 0.95 * W or h > 0.95 * H:
            continue

        ratio = w / float(h + 1e-6)
        if ratio < 1.2:
            continue

        score = area * (1.2 if y < 0.6 * H else 1.0)
        cand.append((score, x, y, w, h))

    cand.sort(reverse=True, key=lambda t: t[0])
    cand = cand[:max_rois]

    out = []
    for i, (_score, x, y, w, h) in enumerate(cand, 1):
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        roi_bgr = imagen_bgr[y1:y2, x1:x2]
        roi_gray = preparar_gray(roi_bgr)
        out.append(RoiCandidate(name=f"TEXTO_OSCURO_{i}", gray=roi_gray))

    return out


def encontrar_caja_sap(gray) -> Optional[Tuple[int, int, int, int]]:
    """
    Busca token 'SAP' (simple y controlado). Devuelve bbox en coordenadas globales.
    """
    H, W = gray.shape[:2]
    x0 = int(0.30 * W)
    roi = gray[0:int(0.60 * H), x0:W]

    data = pytesseract.image_to_data(
        roi, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
    )

    def norm_token(s):
        s = (s or "").upper().strip()
        s = re.sub(r"[^A-Z0-9]", "", s)
        s = s.replace("5", "S").replace("4", "A")
        return s

    best = None
    for txt, x, y, w, h, conf in zip(
        data.get("text", []),
        data.get("left", []),
        data.get("top", []),
        data.get("width", []),
        data.get("height", []),
        data.get("conf", []),
    ):
        if norm_token(txt) == "SAP":
            try:
                c = float(conf) if conf != "-1" else -1.0
            except:
                c = -1.0
            X = x + x0
            Y = y
            if best is None or c > best["conf"]:
                best = {"bbox": (X, Y, w, h), "conf": c}

    return best["bbox"] if best else None


def recortar_roi_numero_despues_de_sap(gray, caja_sap) -> Optional[RoiCandidate]:
    x, y, w, h = caja_sap
    H, W = gray.shape

    x1 = min(W, x + w + int(0.25 * h))
    y1 = max(0, y - int(0.15 * h))
    y2 = min(H, y + int(1.05 * h))
    x2 = min(W, x1 + int(12.5 * h))

    roi = gray[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None
    return RoiCandidate(name="SAP_DERECHA", gray=roi)


def zonas_fallback(imagen_bgr) -> List[RoiCandidate]:
    H, W = imagen_bgr.shape[:2]
    zonas = [
        ("Z_top_right", imagen_bgr[int(0.00*H):int(0.55*H), int(0.40*W):int(1.00*W)]),
        ("Z_left_mid",  imagen_bgr[int(0.20*H):int(0.80*H), int(0.00*W):int(0.60*W)]),
        ("Z_mid",       imagen_bgr[int(0.20*H):int(0.75*H), int(0.20*W):int(0.85*W)]),
    ]
    out = []
    for name, zona in zonas:
        g = preparar_gray(zona)
        out.append(RoiCandidate(name=f"ZONA_{name}", gray=g))
    return out


# =====================================================
# MAIN PIPELINE (base para iterar)
# =====================================================
def obtener_id_producto(
    imagen_bgr,
    conn=None,
    debug: bool = False,
    debug_binarizaciones: bool = False
) -> Tuple[Optional[str], str]:
    """
    Pipeline (ordenado por lo que mejor te funciona):
      1) ETIQUETA_BLANCA (sticker)
      2) TEXTO_OSCURO (black-hat ROIs)
      3) SAP_DERECHA
      4) ZONAS_FALLBACK

    En debug:
      - muestra ROI_USADA_OCR (ganadora)
      - si falla, muestra ROI_MEJOR_INTENTO
    """
    existe_bd = existe_id_en_bd(conn, strict=True)
    candidatos: List[Dict[str, Any]] = []
    mejor_intento = {"roi": None, "conf": -1.0, "name": "N/A", "tag": ""}

    def evaluar_roi(rc: RoiCandidate):
        nonlocal mejor_intento, candidatos

        pid, conf, tag = ocr_votacion_id(rc.gray, existe_bd=existe_bd)
        if conf > mejor_intento["conf"]:
            mejor_intento = {"roi": rc.gray,
                             "conf": conf, "name": rc.name, "tag": tag}

        if pid is None:
            return

        ok_bd = existe_bd(pid)
        candidatos.append({
            "pid": pid,
            "conf": conf,
            "name": rc.name,
            "tag": tag,
            "ok_bd": ok_bd,
            "roi": rc.gray
        })

    # 1) ETIQUETA BLANCA
    rc = encontrar_roi_etiqueta_blanca(imagen_bgr)
    if rc is not None:
        evaluar_roi(rc)

    # 2) TEXTO OSCURO
    for rc in encontrar_rois_por_texto_oscuro(imagen_bgr, max_rois=3):
        evaluar_roi(rc)

    # 3) SAP
    gray_full = preparar_gray(imagen_bgr)
    caja_sap = encontrar_caja_sap(gray_full)
    if caja_sap is not None:
        rc_sap = recortar_roi_numero_despues_de_sap(gray_full, caja_sap)
        if rc_sap is not None:
            evaluar_roi(rc_sap)

    # 4) ZONAS
    for rc in zonas_fallback(imagen_bgr):
        evaluar_roi(rc)

    # =========================
    # Selección final
    # =========================

    if candidatos:
        # score: primero BD, luego votos implícitos (no los exponemos), luego longitud y conf
        # acá simplificamos: BD > len(pid) > conf
        def score(c):
            return (1 if c["ok_bd"] else 0, len(c["pid"]), c["conf"])

        ganador = max(candidatos, key=score)

        if debug:
            mostrar_roi(
                f"ROI_USADA_OCR | {ganador['name']} | {ganador['tag']}", ganador["roi"])
            if debug_binarizaciones:
                mostrar_pack_binarizaciones(
                    f"BINARIZACIONES | {ganador['name']}", ganador["roi"])

        if ganador["ok_bd"]:
            return ganador["pid"], f"OK_{ganador['name']} ({ganador['tag']})"
        else:
            # Si quieres: aquí puedes decidir si rechazas IDs que no validan BD (return None)
            return ganador["pid"], f"ID_OCR_NO_VALIDA_BD_{ganador['name']} ({ganador['tag']})"

    # No hubo candidatos
    if debug and mejor_intento["roi"] is not None:
        mostrar_roi(
            f"ROI_MEJOR_INTENTO | {mejor_intento['name']} | {mejor_intento['tag']}", mejor_intento["roi"])
        if debug_binarizaciones:
            mostrar_pack_binarizaciones(
                f"BINARIZACIONES | MEJOR_INTENTO {mejor_intento['name']}", mejor_intento["roi"])

    return None, "No se pudo detectar ID"


# =====================================================
# TEST
# =====================================================
if __name__ == "__main__":
    ruta_imagen = r"C:\Users\nicol\Downloads\Prueba_etiqueta7.jpeg"
    # ruta_imagen = r"C:\Users\nicol\Downloads\Prueba_etiqueta29.jpg"
    imagen = cv2.imread(ruta_imagen)
    # 11 27 29
    CONN_STR = (
        f"DRIVER={{{os.getenv('DB_DRIVER')}}};"
        f"SERVER={os.getenv('DB_SERVER')};"
        f"DATABASE={os.getenv('DB_NAME')};"
        f"UID={os.getenv('DB_USER')};"
        f"PWD={os.getenv('DB_PASSWORD')};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    conn = pyodbc.connect(CONN_STR)

    pid, info = obtener_id_producto(
        imagen, conn=conn, debug=True, debug_binarizaciones=False)
    print("Resultado:", pid, "|", info)

    if conn is not None:
        conn.close()

    cv2.destroyAllWindows()
