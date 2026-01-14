import json
import paho.mqtt.publish as publish

BROKER_HOST = "localhost"
BROKER_PORT = 1883
BASE_TOPIC = "robot/sensado"


MAPEO_SENALES = {
    "bateria_estado_carga": (1, 5),   # Pieza 1, métrica 5
    "bateria_voltaje":      (1, 4),
    "micro_esclavo_voltaje":       (3, 4),
    "micro_esclavo_temperatura":       (3, 2),
    "micro_maestro_voltaje":       (2, 4),
    "micro_maestro_temperatura":       (2, 2),
    "micro_maestro_velocidad":       (2, 1),
    "ruedas_desviacion":       (4, 3),
    "ultrasonido_distancia":       (6, 6),
    "ultrasonido_velocidad":       (6, 1),
    "Lidar_velocidad":       (5, 1),
    "Lidar_distancia":       (5, 6)
}


def publicar_medicion(signal_name: str, valor: float):
    if signal_name not in MAPEO_SENALES:
        raise ValueError(f"Señal desconocida: {signal_name}")

    id_pieza, id_metrica = MAPEO_SENALES[signal_name]

    valores = {
        "id_pieza": id_pieza,
        "id_metrica": id_metrica,
        "valor": float(valor)
    }
    topic = f"{BASE_TOPIC}/{id_pieza}/{id_metrica}"
    publish.single(
        topic,
        payload=json.dumps(valores),
        hostname=BROKER_HOST,
        port=BROKER_PORT
    )
    print("Publicado en:", topic, "|", valores)


def main():
    signal = "bateria_estado_carga"
    lectura = 78.0
    publicar_medicion(signal, lectura)


if __name__ == "__main__":
    main()
