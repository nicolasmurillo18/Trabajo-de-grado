import json
import pyodbc
import paho.mqtt.client as mqtt
import os
from dotenv import load_dotenv

load_dotenv()  # Carga el .env

MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC_SUB = "robot/sensado/+/+"  # robot/sensado/<id_pieza>/<id_metrica>

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


def parse_topic(topic: str):
    secciones = topic.split("/")  # partes del tópico
    if len(secciones) != 4:
        raise ValueError(f"Tópico inválido: {topic}")
    _, _, id_pieza, id_metrica = secciones
    return int(id_pieza), int(id_metrica)


def parse_value(valor: bytes):
    decodificacion = valor.decode("utf-8").strip()

    # La condición para verificar si viene en el formato del publisher
    if decodificacion.startswith("{"):
        obj = json.loads(decodificacion)
        return float(obj["valor"])  # Retorna el valor del publisher

    return float(decodificacion.replace(",", "."))


def insertar_sensado(conn, id_pieza: int, id_metrica: int, valor: float):
    cursor = conn.cursor()
    query = """
        INSERT INTO rob.Sensado (Id_pieza, Id_metrica, Fecha_medicion, Valor)
        VALUES (?, ?, SYSDATETIME(), ?);
    """
    cursor.execute(query, (id_pieza, id_metrica, valor))
    conn.commit()


def on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        print(f"Conectado a MQTT en {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe(TOPIC_SUB)
        print(f"Suscrito a: {TOPIC_SUB}")
    else:
        print(f"No se pudo conectar. Código: {rc}")


def on_message(client, userdata, mensaje):
    conn = userdata["conn"]
    try:
        id_pieza, id_metrica = parse_topic(mensaje.topic)
        valor = parse_value(mensaje.payload)

        insertar_sensado(conn, id_pieza, id_metrica, valor)

        print(f"[INSERTADO] {mensaje.topic} -> valor={valor}")
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] {mensaje.topic} | {e}")


def main():
    conn = pyodbc.connect(CONN_STR)
    client = mqtt.Client()
    client.user_data_set({"conn": conn})

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
