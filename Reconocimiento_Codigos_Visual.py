import cv2
import pytesseract
import re
import pyodbc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
import os
import numpy as np

from inventory_publisher import publicar_inventario

load_dotenv()

# =====================================================
# CONFIGURACIÓN OCR (Tesseract)
# =====================================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Config numérica: solo dígitos, optimizado para IDs 5-8 dígitos.
OCR_NUM_CONFIG = (
    "--oem 3 "
    "--dpi 300 "
    "-c user_defined_dpi=300 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c classify_bln_numeric_mode=1 "
    "-c load_system_dawg=0 -c load_freq_dawg=0 "
)

# PSMs a evaluar en la etapa pesada (más cobertura, más costo).
OCR_PSMS = (7, 8, 6, 13)

# Detecta explícitamente "SAP ... <número>" en texto libre (sin whitelist).
REGEX_SAP = re.compile(r"SAP\D{0,10}(\d{5,8})", re.IGNORECASE)

# =====================================================
# VALIDACIÓN EN BD (con caché)
# =====================================================


def existe_id_en_bd(conn, strict: bool = True) -> Callable[[str], bool]:
    """
    Retorna una función cerrada (closure) que valida si un Id_producto existe en BD.
    Incluye caché para evitar repetir SELECTs cuando el OCR produce candidatos repetidos.

    strict=True:
      - si no hay conexión o hay error => False (no valida)
    strict=False:
      - si no hay conexión o hay error => True (modo permisivo)
    """
    cache: Dict[str, bool] = {}

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
        except Exception:
            ok = False if strict else True

        cache[pid] = ok
        return ok

    return _existe

# =====================================================
# HEURÍSTICAS: FILTRO DE “BASURA”
# =====================================================


def es_candidato_basura(pid: str) -> bool:
    """
    Filtra falsos positivos típicos:
    - EAN13 que se cuelan (códigos largos de barras)
    - Fechas compactas en formato ddmmyyyy o yyyymmdd
    """
    # EAN-13 típico (si se cuela)
    if len(pid) == 13 and pid.startswith(("770", "777", "778", "779")):
        return True

    # Fechas comprimidas a 8 dígitos
    if len(pid) == 8:
        dd = int(pid[0:2])
        mm = int(pid[2:4])
        yyyy = int(pid[4:8])
        if 1 <= dd <= 31 and 1 <= mm <= 12 and 2000 <= yyyy <= 2099:
            return True

        yyyy = int(pid[0:4])
        mm = int(pid[4:6])
        dd = int(pid[6:8])
        if 2000 <= yyyy <= 2099 and 1 <= mm <= 12 and 1 <= dd <= 31:
            return True

    return False

# =====================================================
# PREPROCESAMIENTO (imagen -> gray, normalización, binarizaciones)
# =====================================================


def preparar_gray(imagen_bgr, target: int = 1600):
    """
    Convierte a gris + aplica CLAHE para mejorar contraste.
    Además, escala hacia arriba si la imagen es pequeña (mejora OCR).
    """
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


def recorte_bordes(gray, pct: float = 0.02):
    """
    Recorta un borde pequeño para eliminar marcos/sombras laterales que confunden OCR.
    """
    if gray is None or gray.size == 0:
        return gray
    h, w = gray.shape[:2]
    dx = max(1, int(pct * w))
    dy = max(1, int(pct * h))
    if w - 2 * dx <= 5 or h - 2 * dy <= 5:
        return gray
    return gray[dy:h - dy, dx:w - dx]


def preparar_para_ocr(gray):
    """
    Acondiciona el ROI antes del OCR:
    - Escala si es muy pequeño
    - Suaviza (reduce ruido) + sharpen controlado
    - Normaliza contraste
    """
    if gray is None or gray.size == 0:
        return gray

    h, _w = gray.shape[:2]
    if h < 220:
        s = min(1.8, max(1.0, 220 / float(h)))
        if s > 1.05:
            gray = cv2.resize(gray, None, fx=s, fy=s,
                              interpolation=cv2.INTER_CUBIC)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def binarizaciones(gray_roi, etapa: int = 1):
    """
    Genera variantes binarias del ROI:
    - etapa 1: variantes baratas (Otsu + invertido + dilatación)
    - etapa 2: agrega variantes más agresivas (adaptive, sharpen, norm)
    """
    if gray_roi is None or gray_roi.size == 0:
        return []

    out = []

    g = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(th)
    out.append(cv2.bitwise_not(th))

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th_d = cv2.dilate(th, k, iterations=1)
    out.append(th_d)
    out.append(cv2.bitwise_not(th_d))

    if etapa == 1:
        return out

    th2 = cv2.adaptiveThreshold(
        gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )
    out.append(th2)
    out.append(cv2.bitwise_not(th2))

    blur = cv2.GaussianBlur(gray_roi, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray_roi, 1.6, blur, -0.6, 0)
    g2 = cv2.GaussianBlur(sharp, (3, 3), 0)
    _, th3 = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(th3)
    out.append(cv2.bitwise_not(th3))

    norm = cv2.normalize(gray_roi, None, 0, 255, cv2.NORM_MINMAX)
    gn = cv2.GaussianBlur(norm, (3, 3), 0)
    _, th4 = cv2.threshold(gn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(th4)
    out.append(cv2.bitwise_not(th4))

    return out


def score_barcode_like(gray_roi) -> float:
    """
    Mide si un ROI “se parece” a un código de barras:
    - muchas transiciones verticales (gradiente en X)
    - densidad de bordes + intensidad promedio
    Se usa para decidir si añadimos PSMs extra (11/12) en etapa 2.
    """
    if gray_roi is None or gray_roi.size == 0:
        return 0.0

    h, w = gray_roi.shape[:2]
    if w > 700:
        s = 700 / float(w)
        gray_roi = cv2.resize(
            gray_roi, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    gx = cv2.Sobel(gray_roi, cv2.CV_16S, 1, 0, ksize=3)
    absx = cv2.convertScaleAbs(gx)

    edges = (absx > 40).astype("uint8")
    dens = edges.mean()
    mean_int = float(gray_roi.mean()) / 255.0

    return (0.65 * dens) + (0.35 * mean_int)

# =====================================================
# OCR CORE: extracción robusta de candidatos y votación
# =====================================================


def _extraer_candidatos_desde_data(data: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    A partir del output de image_to_data(), extrae candidatos numéricos 5-8 dígitos.
    Considera:
      - tokens individuales
      - concatenación de tokens en una misma línea (para casos donde Tesseract separa dígitos)
    Devuelve lista ordenada de (pid, conf).
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
        except Exception:
            conf = -1.0

        ln = data.get("line_num", [0] * n)[i]
        left = data.get("left", [0] * n)[i]
        items.append((ln, left, d, conf))

    if not items:
        return []

    # Orden visual aproximado: línea y posición horizontal
    items.sort(key=lambda x: (x[0], x[1]))

    por_linea: Dict[int, List[Tuple[int, str, float]]] = {}
    for ln, left, d, conf in items:
        por_linea.setdefault(ln, []).append((left, d, conf))

    cands: List[Tuple[str, float]] = []

    # Tokens 5-8
    for ln in por_linea:
        for _left, d, conf in por_linea[ln]:
            if 5 <= len(d) <= 8:
                cands.append((d, conf))

    # Concatenaciones 5-8 dentro de la misma línea
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

    # Consolidar por pid: conservar mayor confianza
    best: Dict[str, float] = {}
    for pid, conf in cands:
        if pid not in best or conf > best[pid]:
            best[pid] = conf

    out = [(pid, best[pid]) for pid in best.keys()]
    out.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
    return out


def ocr_votacion_id(gray_roi, existe_bd: Optional[Callable[[str], bool]] = None) -> Tuple[Optional[str], float, str]:
    """
    OCR escalonado (rápido -> medio -> pesado):
      NIVEL 0: busca "SAP ... <id>" en la parte superior
      NIVEL 1: OCR numérico simple (psm 7/8, Otsu)
      NIVEL 2: Votación etapa1 medio (pocas combinaciones)
      NIVEL 3: Votación pesada (etapa 1 completa + etapa 2 completa)
    """
    if gray_roi is None or gray_roi.size == 0:
        return None, -1.0, "EMPTY_ROI"

    gray_roi = recorte_bordes(gray_roi, pct=0.02)

    # CAP ancho para limitar tiempo de OCR en ROIs enormes
    h0, w0 = gray_roi.shape[:2]
    max_w = 1100
    if w0 > max_w:
        s = max_w / float(w0)
        gray_roi = cv2.resize(
            gray_roi, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)

    H = gray_roi.shape[0]
    roi_top = gray_roi[: int(0.45 * H), :]

    # -------------------------
    # NIVEL 0: FAST SAP
    # -------------------------
    try:
        sap_txt = pytesseract.image_to_string(
            roi_top, config="--oem 3 --psm 6")
        m = REGEX_SAP.search(sap_txt or "")
        if m:
            pid = m.group(1)
            if not es_candidato_basura(pid) and (existe_bd is None or existe_bd(pid)):
                return pid, 85.0, "TOP/FAST_SAP_psm6"
    except Exception:
        pass

    # -------------------------
    # NIVEL 1: FAST numérico
    # -------------------------
    def _fast_try(base_img, region_tag: str) -> Tuple[Optional[str], float, str]:
        if base_img is None or base_img.size == 0:
            return None, -1.0, ""

        base_img = preparar_para_ocr(base_img)
        g = cv2.GaussianBlur(base_img, (3, 3), 0)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_inv = cv2.bitwise_not(th)

        for img_bin in (th, th_inv):
            for psm in (7, 8):
                txt = pytesseract.image_to_string(
                    img_bin, config=f"{OCR_NUM_CONFIG} --psm {psm}")
                d = re.sub(r"\D", "", txt or "")
                m2 = re.search(r"(\d{5,8})", d)
                if not m2:
                    continue
                pid = m2.group(1)
                if es_candidato_basura(pid):
                    continue
                if existe_bd is None or existe_bd(pid):
                    return pid, 80.0, f"{region_tag}/FAST_psm{psm}"

        return None, -1.0, ""

    for base_img, region_tag in ((roi_top, "TOP"), (gray_roi, "FULL")):
        pid_fast, conf_fast, tag_fast = _fast_try(base_img, region_tag)
        if pid_fast is not None:
            return pid_fast, conf_fast, tag_fast

    # -------------------------
    # NIVEL 2/3: votación
    # -------------------------
    votos: Dict[str, Dict[str, Any]] = {}

    def _add_vote(pid: str, conf: float, tag: str):
        if pid not in votos:
            votos[pid] = {"votes": 0, "best_conf": conf, "tag": tag}
        votos[pid]["votes"] += 1
        if conf > votos[pid]["best_conf"]:
            votos[pid]["best_conf"] = conf
            votos[pid]["tag"] = tag

    def _try_etapa1_medio() -> Optional[Tuple[str, float, str]]:
        psms_local = (7, 8)

        for base_img, region_tag in ((roi_top, "TOP"), (gray_roi, "FULL")):
            if base_img is None or base_img.size == 0:
                continue

            base_img = recorte_bordes(base_img, pct=0.02)
            base_img = preparar_para_ocr(base_img)

            g = cv2.GaussianBlur(base_img, (3, 3), 0)
            _, th = cv2.threshold(
                g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th_inv = cv2.bitwise_not(th)

            for thx, th_tag in ((th, "OTSU"), (th_inv, "OTSU_INV")):
                for psm in psms_local:
                    data = pytesseract.image_to_data(
                        thx,
                        config=f"{OCR_NUM_CONFIG} --psm {psm}",
                        output_type=pytesseract.Output.DICT
                    )
                    cands = _extraer_candidatos_desde_data(data)
                    if not cands:
                        continue

                    for pid, conf in cands[:2]:
                        if es_candidato_basura(pid):
                            continue
                        tag = f"{region_tag}/E1M_{th_tag}_psm{psm}"
                        _add_vote(pid, conf, tag)
                        if existe_bd is not None and existe_bd(pid):
                            return pid, float(votos[pid]["best_conf"]), votos[pid]["tag"]
        return None

    early = _try_etapa1_medio()
    if early is not None:
        return early

    def _try_etapa_pesada(etapa: int) -> Optional[Tuple[str, float, str]]:
        for base_img, region_tag in ((roi_top, "TOP"), (gray_roi, "FULL")):
            base_img = recorte_bordes(base_img, pct=0.02)
            base_img = preparar_para_ocr(base_img)

            psms_local = (7, 8) if etapa == 1 else OCR_PSMS

            # Si parece barcode, habilitamos psm 11/12 (mejora casos difíciles)
            if etapa == 2:
                bscore = score_barcode_like(base_img)
                if bscore > 0.12:
                    psms_local = tuple(list(psms_local) + [11, 12])

            for th in binarizaciones(base_img, etapa=etapa):
                for psm in psms_local:
                    data = pytesseract.image_to_data(
                        th,
                        config=f"{OCR_NUM_CONFIG} --psm {psm}",
                        output_type=pytesseract.Output.DICT
                    )
                    cands = _extraer_candidatos_desde_data(data)
                    if not cands:
                        continue

                    for pid, conf in cands[:2]:
                        if es_candidato_basura(pid):
                            continue
                        tag = f"{region_tag}/E{etapa}_psm{psm}"
                        _add_vote(pid, conf, tag)
                        if existe_bd is not None and existe_bd(pid):
                            return pid, float(votos[pid]["best_conf"]), votos[pid]["tag"]
        return None

    early = _try_etapa_pesada(etapa=1)
    if early is not None:
        return early

    early = _try_etapa_pesada(etapa=2)
    if early is not None:
        return early

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

    # Control de evidencia mínima si no validó BD
    if existe_bd is not None:
        bd_ok = bool(existe_bd(ganador))
        if not bd_ok and (best_votes < 2 or best_conf < 35):
            return None, best_conf, f"LOW_EVIDENCE({best_tag})"

    return ganador, best_conf, best_tag

# =====================================================
# ROI CANDIDATES (estructura estándar para ROIs en gris)
# =====================================================


@dataclass
class RoiCandidate:
    name: str
    gray: Any
    meta: str = ""

# =====================================================
# DETECCIÓN DE ROIs (etiqueta blanca, etiqueta por rectángulo, barcode, texto)
# =====================================================


def encontrar_roi_etiqueta_rect(imagen_bgr) -> Optional[RoiCandidate]:
    """
    Detecta sticker por geometría (rectángulo convexo) + interior claro + borde/contenido.
    Este método ayuda especialmente en fotos difíciles donde HSV/LAB falla.
    """
    H_img, W_img = imagen_bgr.shape[:2]
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 60, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        area_rel = area / float(H_img * W_img)

        if area_rel < 0.008 or area_rel > 0.35:
            continue

        ratio = w / float(h + 1e-6)
        if ratio < 1.1 or ratio > 10.0:
            continue

        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        mean_int = roi.mean()
        if mean_int < 130:
            continue

        e = cv2.Canny(roi, 60, 160).mean() / 255.0
        if e < 0.010:
            continue

        score = area * (1.2 if y < 0.75 * H_img else 1.0)
        if best is None or score > best["score"]:
            best = {"x": x, "y": y, "w": w, "h": h,
                    "score": score, "e": e, "mean": mean_int}

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
    return RoiCandidate(name="ETIQUETA_RECT", gray=roi_gray, meta=f"edge={best['e']:.4f},mean={best['mean']:.1f}")


def encontrar_rois_etiqueta_blanca(imagen_bgr, max_rois: int = 3) -> List[RoiCandidate]:
    """
    Detecta áreas blancas tipo sticker con HSV + LAB (más tolerante ante iluminación).
    Devuelve top-N ROIs por score geométrico + borde + contraste con entorno.
    """
    H_img, W_img = imagen_bgr.shape[:2]
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)

    mask_hsv = cv2.inRange(hsv, (0, 0, 135), (180, 120, 255))
    L, _A, _B = cv2.split(lab)
    mask_lab = cv2.inRange(L, 150, 255)
    mask = cv2.bitwise_and(mask_hsv, mask_lab)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=2
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    cand = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        area_rel = area / float(H_img * W_img)

        if area_rel < 0.006 or area_rel > 0.40:
            continue

        ratio = w / float(h + 1e-6)
        if ratio < 1.1 or ratio > 10.0:
            continue

        contour_area = cv2.contourArea(c)
        if contour_area <= 0:
            continue

        rectangularidad = contour_area / float(area + 1e-6)
        if rectangularidad < 0.70:
            continue

        hull = cv2.convexHull(c)
        solidez = contour_area / float(cv2.contourArea(hull) + 1e-6)
        if solidez < 0.85:
            continue

        roi = imagen_bgr[y:y+h, x:x+w]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        edge_density = (cv2.Canny(g, 60, 160).mean() / 255.0)
        if edge_density < 0.006:
            continue

        pad_ring = 18
        x1 = max(0, x - pad_ring)
        y1 = max(0, y - pad_ring)
        x2 = min(W_img, x + w + pad_ring)
        y2 = min(H_img, y + h + pad_ring)
        ring = cv2.cvtColor(imagen_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        inside_mean = float(g.mean())
        ring_mean = float(ring.mean())
        delta = inside_mean - ring_mean

        pos_bonus = 1.0
        if 0.20 * H_img < y < 0.80 * H_img:
            pos_bonus *= 1.15
        if x < 0.70 * W_img:
            pos_bonus *= 1.10

        score = area * (1.0 + 2.0 * edge_density) * \
            (1.0 + max(0.0, delta) / 60.0) * pos_bonus
        cand.append((score, x, y, w, h, edge_density, delta))

    if not cand:
        return []

    cand.sort(reverse=True, key=lambda t: t[0])
    cand = cand[:max_rois]

    out = []
    pad = 12
    for _score, x, y, w, h, ed, delta in cand:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W_img, x + w + pad)
        y2 = min(H_img, y + h + pad)

        roi_bgr = imagen_bgr[y1:y2, x1:x2]
        roi_gray = preparar_gray(roi_bgr)
        out.append(RoiCandidate(name="ETIQUETA_BLANCA", gray=roi_gray,
                   meta=f"edge={ed:.4f},delta={delta:.1f}"))

    return out


def subrois_etiqueta_blanca(rc: RoiCandidate) -> List[RoiCandidate]:
    """
    Sub-ROIs para aislar el número grande del sticker.
    En 27/28 suele estar arriba-izquierda y se contamina con EAN/fechas.
    """
    g = rc.gray
    if g is None or g.size == 0:
        return []

    H, W = g.shape[:2]
    out = []

    id_big = g[int(0.00 * H):int(0.45 * H), int(0.00 * W):int(0.70 * W)]
    out.append(RoiCandidate(name=rc.name + "_IDBIG", gray=id_big, meta=rc.meta))

    top = g[int(0.00 * H):int(0.50 * H), :]
    out.append(RoiCandidate(name=rc.name + "_TOP", gray=top, meta=rc.meta))

    left_mid = g[int(0.10 * H):int(0.70 * H), int(0.00 * W):int(0.55 * W)]
    out.append(RoiCandidate(name=rc.name + "_LEFTMID",
               gray=left_mid, meta=rc.meta))

    return out


def zonas_fallback(imagen_bgr) -> List[RoiCandidate]:
    """
    Zonas “genéricas” por si fallan las detecciones de etiqueta/barcode/texto.
    Son recortes amplios que a veces capturan IDs grandes en contextos raros.
    """
    H, W = imagen_bgr.shape[:2]
    zonas = [
        ("Z_top_right", imagen_bgr[int(0.00 * H):int(0.55 * H), int(0.40 * W):int(1.00 * W)]),
        ("Z_left_mid",  imagen_bgr[int(0.20 * H):int(0.80 * H), int(0.00 * W):int(0.60 * W)]),
        ("Z_mid",       imagen_bgr[int(0.20 * H):int(0.75 * H), int(0.20 * W):int(0.85 * W)]),
        ("Z_center",        imagen_bgr[int(
            0.20 * H):int(0.80 * H), int(0.10 * W):int(0.90 * W)]),
        ("Z_center_lower",  imagen_bgr[int(
            0.35 * H):int(0.88 * H), int(0.10 * W):int(0.90 * W)]),
    ]
    out = []
    for name, zona in zonas:
        g = preparar_gray(zona)
        out.append(RoiCandidate(name=f"ZONA_{name}", gray=g))
    return out


def encontrar_roi_arriba_de_barcode(imagen_bgr) -> Optional[RoiCandidate]:
    """
    Detecta un bloque tipo barcode (barras verticales) y recorta una franja superior
    para capturar el ID impreso cerca del código de barras.
    """
    if imagen_bgr is None or imagen_bgr.size == 0:
        return None

    H_img, W_img = imagen_bgr.shape[:2]
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    scale = 1.0
    if W_img > 1600:
        scale = 1600 / float(W_img)
        gray_s = cv2.resize(
            gray, (int(W_img * scale), int(H_img * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray_s = gray

    gx = cv2.Scharr(gray_s, cv2.CV_16S, 1, 0)
    absx = cv2.convertScaleAbs(gx)

    absx = cv2.GaussianBlur(absx, (5, 5), 0)
    _, bw = cv2.threshold(absx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 9))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=2)

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open, iterations=1)

    contours, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    Hs, Ws = gray_s.shape[:2]
    best = None

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        area_rel = area / float(Hs * Ws)

        if area_rel < 0.0015 or area_rel > 0.25:
            continue

        ratio = w / float(h + 1e-6)
        if ratio < 2.2:
            continue
        if h < 18:
            continue

        score = area * (1.35 if y > 0.35 * Hs else 1.0)
        if best is None or score > best["score"]:
            best = {"x": x, "y": y, "w": w, "h": h, "score": score}

    if best is None:
        return None

    x = int(best["x"] / scale)
    y = int(best["y"] / scale)
    w = int(best["w"] / scale)
    h = int(best["h"] / scale)

    y1 = max(0, y - int(2.2 * h))
    y2 = min(H_img, y + int(0.15 * h))
    x1 = max(0, x - int(0.08 * w))
    x2 = min(W_img, x + w + int(0.08 * w))

    roi_bgr = imagen_bgr[y1:y2, x1:x2]
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    roi_gray = preparar_gray(roi_bgr)
    return RoiCandidate(name="ARRIBA_BARCODE", gray=roi_gray, meta=f"bbox=({x1},{y1},{x2},{y2})")


def encontrar_rois_por_texto_oscuro(imagen_bgr, max_rois: int = 3) -> List[RoiCandidate]:
    """
    Busca bloques con “texto oscuro sobre fondo claro” usando blackhat.
    Útil cuando la etiqueta blanca no existe o el ID está impreso directo en cartón.
    """
    gray0 = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray0.shape[:2]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    blackhat = cv2.morphologyEx(gray0, cv2.MORPH_BLACKHAT, k)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    _, bw = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 11))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k2, iterations=2)

    contours, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    geom = []
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

        roi_gray_tmp = gray0[y:y + h, x:x + w]
        if roi_gray_tmp.mean() < 90:
            continue

        score_geom = area * (1.2 if y < 0.6 * H else 1.0)
        geom.append((score_geom, x, y, w, h))

    if not geom:
        return []

    geom.sort(reverse=True, key=lambda t: t[0])
    top_k = geom[: max(10, max_rois * 4)]

    cand = []
    for score_geom, x, y, w, h in top_k:
        roi_gray_tmp = gray0[y:y + h, x:x + w]
        barcode_score = score_barcode_like(roi_gray_tmp)
        score = score_geom * (1.0 + 1.6 * barcode_score)
        cand.append((score, x, y, w, h))

    cand.sort(reverse=True, key=lambda t: t[0])
    cand = cand[:max_rois]

    out = []
    pad = 10
    for i, (_score, x, y, w, h) in enumerate(cand, 1):
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        roi_bgr = imagen_bgr[y1:y2, x1:x2]
        roi_gray = preparar_gray(roi_bgr)
        out.append(RoiCandidate(name=f"TEXTO_OSCURO_{i}", gray=roi_gray))
    return out


def encontrar_caja_sap_fast(gray) -> Optional[Tuple[int, int, int, int]]:
    """
    Heurística: detecta presencia de la palabra SAP en una zona probable.
    Si aparece, construye una 'caja' aproximada donde debería estar el número.
    """
    H, W = gray.shape[:2]
    x0 = int(0.30 * W)
    roi = gray[0:int(0.60 * H), x0:W]

    txt = pytesseract.image_to_string(roi, config="--oem 3 --psm 6")
    t = re.sub(r"[^A-Z0-9]", "", (txt or "").upper()
               ).replace("5", "S").replace("4", "A")
    if "SAP" not in t:
        return None

    x = x0 + int(0.05 * (W - x0))
    y = int(0.10 * (0.60 * H))
    w = int(0.20 * (W - x0))
    h = int(0.15 * (0.60 * H))
    return (x, y, w, h)


def recortar_roi_numero_despues_de_sap(gray, caja_sap) -> Optional[RoiCandidate]:
    """
    Dada la caja aproximada del texto 'SAP', recorta a la derecha donde suele venir el ID.
    """
    x, y, w, h = caja_sap
    H, W = gray.shape[:2]

    x1 = min(W, x + w + int(0.25 * h))
    y1 = max(0, y - int(0.15 * h))
    y2 = min(H, y + int(1.05 * h))
    x2 = min(W, x1 + int(12.5 * h))

    roi = gray[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None
    return RoiCandidate(name="SAP_DERECHA", gray=roi)

# =====================================================
# PIPELINE PRINCIPAL: obtiene ID desde imagen + valida contra BD
# =====================================================


def obtener_id_producto(imagen_bgr, conn=None) -> Tuple[Optional[str], str]:
    """
    Orquestador principal:
    - Genera ROIs candidatos (etiqueta blanca/rectángulo/barcode/texto/SAP/zonas)
    - Para cada ROI llama OCR escalonado
    - Valida en BD y retorna la primera coincidencia confiable
    """
    existe_bd = existe_id_en_bd(conn, strict=True)
    candidatos: List[Dict[str, Any]] = []

    def evaluar_roi(rc: RoiCandidate) -> Optional[Tuple[str, str]]:
        pid, conf, tag = ocr_votacion_id(rc.gray, existe_bd=existe_bd)
        if pid is None:
            return None

        # Nota: esto lo mantengo porque dijiste que ya te funciona así.
        ok_bd = True if ("OK_" in tag or "FAST_SAP" in tag or "FAST" in tag) else bool(
            existe_bd(pid))

        candidatos.append(
            {"pid": pid, "conf": conf, "name": rc.name, "tag": tag, "ok_bd": ok_bd})

        if ok_bd:
            return pid, f"OK_{rc.name} ({tag})"
        return None

    # 1) Etiqueta blanca (top-N) + sub-ROIs (clave para 27/28)
    for rc in encontrar_rois_etiqueta_blanca(imagen_bgr, max_rois=2):
        out = evaluar_roi(rc)
        if out is not None:
            return out
        for sub in subrois_etiqueta_blanca(rc):
            out = evaluar_roi(sub)
            if out is not None:
                return out

    # 2) Etiqueta por rectángulo (robusto si falla el blanco)
    rc = encontrar_roi_etiqueta_rect(imagen_bgr)
    if rc is not None:
        out = evaluar_roi(rc)
        if out is not None:
            return out

    # 3) ROI arriba del barcode (IDs cercanos a código de barras)
    rc = encontrar_roi_arriba_de_barcode(imagen_bgr)
    if rc is not None:
        out = evaluar_roi(rc)
        if out is not None:
            return out

    # 4) Texto oscuro (cartón con texto negro)
    for rc in encontrar_rois_por_texto_oscuro(imagen_bgr, max_rois=3):
        out = evaluar_roi(rc)
        if out is not None:
            return out

    # 5) SAP (heurística específica)
    gray_full = preparar_gray(imagen_bgr)
    caja_sap = encontrar_caja_sap_fast(gray_full)
    if caja_sap is not None:
        rc_sap = recortar_roi_numero_despues_de_sap(gray_full, caja_sap)
        if rc_sap is not None:
            out = evaluar_roi(rc_sap)
            if out is not None:
                return out

    # 6) Zonas fallback
    for rc in zonas_fallback(imagen_bgr):
        out = evaluar_roi(rc)
        if out is not None:
            return out

    # Si nada validó BD pero hubo OCR, devolvemos el mejor candidato “observable”
    if candidatos:
        ganador = max(candidatos, key=lambda c: (
            1 if c["ok_bd"] else 0, len(c["pid"]), c["conf"]))
        if ganador["ok_bd"]:
            return ganador["pid"], f"OK_{ganador['name']} ({ganador['tag']})"
        return ganador["pid"], f"ID_OCR_NO_VALIDA_BD_{ganador['name']} ({ganador['tag']})"

    return None, "No se pudo detectar ID"

# =====================================================
# UTILIDAD: bytes -> imagen BGR (robot)
# =====================================================


def decode_image_bytes_to_bgr(image_bytes: bytes) -> Optional[Any]:
    """
    Para cuando el robot envíe la foto como bytes (jpg/png):
      - np.frombuffer -> cv2.imdecode -> imagen BGR
    """
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# =====================================================
# TEST
# =====================================================
if __name__ == "__main__":
    ruta_imagen = r"C:\Users\nicol\Downloads\Prueba_etiqueta30.jpeg"  # 33 42 45 50
    imagen = cv2.imread(ruta_imagen)

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

    pid, info = obtener_id_producto(imagen, conn=conn)
    print("Resultado:", pid, "|", info)
    cantidad = 3
    insertado = publicar_inventario(conn, pid, cantidad)
    print("Inventario actualizado:", insertado)

    conn.close()
