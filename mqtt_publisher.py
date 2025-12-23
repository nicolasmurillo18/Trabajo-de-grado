import json
import paho.mqtt.publish as publish

BROKER_HOST = "localhost"
BROKER_PORT = 1883
BASE_TOPIC = "robot/sensado"


MAPEO_SENALES = {
    "bateria_estado_carga": (1, 5),   # Pieza 1, métrica 5
    "bateria_voltaje":      (1, 6),   # Pieza 1, métrica 6
    "micro3_voltaje":       (3, 4),   # Pieza 3, métrica 4
    # agregar las demás
}


def publicar_medicion(signal_name: str, valor: float):
    if signal_name not in MAPEO_SENALES:
        raise ValueError(
            f"Señal desconocida: {signal_name}. Debes mapearla en MAPEO_SENALES.")

    id_pieza, id_metrica = MAPEO_SENALES[signal_name]

    msg = {
        "id_pieza": id_pieza,
        "id_metrica": id_metrica,
        "valor": float(valor)
    }
    topic = f"{BASE_TOPIC}/{id_pieza}/{id_metrica}"
    publish.single(
        topic,
        payload=json.dumps(msg),
        hostname=BROKER_HOST,
        port=BROKER_PORT
    )
    print("Publicado en:", topic, "|", msg)


def main():
    signal = "bateria_estado_carga"
    lectura = 78.0
    publicar_medicion(signal, lectura)


if __name__ == "__main__":
    main()
