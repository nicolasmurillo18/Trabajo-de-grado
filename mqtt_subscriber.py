import json
import pyodbc
import paho.mqtt.client as mqtt

MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC_SUB = "robot/sensado/+/+"
# robot/sensado/<id_pieza>/<id_metrica>

CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=sqlceaiotserver.database.windows.net;"
    "DATABASE=SQLCEAIOTDB;"
    "UID=ROBOTINV;"
    "PWD=Inventario2026*;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)


def get_connection():
    return pyodbc.connect(CONN_STR)


def parse_topic(topic: str):
    parts = topic.split("/")
    if len(parts) != 4:
        raise ValueError(f"Tópico inválido: {topic}")
    _, _, id_pieza, id_metrica = parts
    return int(id_pieza), int(id_metrica)


def parse_value(payload: bytes):
    text = payload.decode("utf-8").strip()

    if text.startswith("{"):
        obj = json.loads(text)
        return float(obj["valor"])

    return float(text.replace(",", "."))


def insertar_sensado(conn, id_pieza: int, id_metrica: int, valor: float):
    cursor = conn.cursor()
    query = """
        INSERT INTO rob.Sensado (Id_pieza, Id_metrica, Fecha_medicion, Valor)
        VALUES (?, ?, SYSDATETIME(), ?);
    """
    cursor.execute(query, (id_pieza, id_metrica, valor))
    conn.commit()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Conectado a MQTT en {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe(TOPIC_SUB)
        print(f"Suscrito a: {TOPIC_SUB}")
    else:
        print(f"No se pudo conectar. Código: {rc}")


def on_message(client, userdata, msg):
    conn = userdata["conn"]
    try:
        id_pieza, id_metrica = parse_topic(msg.topic)
        valor = parse_value(msg.payload)

        insertar_sensado(conn, id_pieza, id_metrica, valor)

        print(f"[OK] {msg.topic} -> valor={valor}")
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] {msg.topic} | {e}")


def main():
    conn = get_connection()
    client = mqtt.Client()
    client.user_data_set({"conn": conn})

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
