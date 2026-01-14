import pyodbc
from typing import Optional


def publicar_inventario(conn: pyodbc.Connection, pid: Optional[str], cantidad: int) -> bool:
    """
    Publica inventario usando el Id_producto.
    Retorna True si se actualizó/inserto algo.
    """

    try:
        cur = conn.cursor()
        cur.execute(
            """
                INSERT INTO rob.Inventario (Id_producto, Cantidad)
                VALUES (?, ?);
                """,
            (int(pid), cantidad),
        )
        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] No se pudo publicar inventario para el id={pid}: {e}")
        return False
