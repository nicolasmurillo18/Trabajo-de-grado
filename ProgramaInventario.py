from datetime import datetime
from datetime import date
import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()  # Carga el .env

# =========================
# Configuración de conexión
# =========================

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

# =========================
# Utilidades de impresión
# =========================


def imprimir_producto(row):
    try:
        print(
            f"- Id: {row.Id_producto} | "
            f"Marca: {row.Marca} | "
            f"Categoría: {row.Categoria} | "
            f"Estado: {row.Estado} | "
            f"Nombre: {row.Nombre_producto}"
        )
    except Exception:
        print(f"- {row}")


def pause():
    input("\nPresiona Enter para continuar...")

# =========================
# Opciones del menú
# =========================


def buscar_por_categoria(conn):
    categoria = input(
        "Escribe la categoría exacta (ej: Aseo Caja, Aseo Bolsa, Alimento Caja): ").strip()
    if not categoria:
        print("Categoría vacía. Cancelado.")
        return

    cursor = conn.cursor()
    query = """
        SELECT *
        FROM rob.Producto
        WHERE Categoria = ?
        ORDER BY Marca, Nombre_producto;
    """
    cursor.execute(query, (categoria,))
    rows = cursor.fetchall()

    print(
        f"\nEn la categoria '{categoria}' hay {len(rows)} productos\n")
    for r in rows:
        imprimir_producto(r)


def buscar_por_nombre(conn):
    nombre = input(
        "Escribe parte del nombre del producto (ej: TALCO, MAIZENA): ").strip()
    if not nombre:
        print("Nombre vacío. Cancelado.")
        return

    cursor = conn.cursor()
    query = """
        SELECT *
        FROM rob.Producto
        WHERE Nombre_producto LIKE ?
        ORDER BY Marca, Nombre_producto;
    """
    cursor.execute(query, (f"%{nombre}%",))
    rows = cursor.fetchall()

    print(
        f"\nLos productos que contienen el nombre '{nombre}' son: {len(rows)}\n")
    for r in rows:
        imprimir_producto(r)


def buscar_por_id(conn):
    raw = input("Escribe el Id_producto: ").strip()
    if not raw.isdigit():
        print("Id inválido. Debe ser numérico.")
        return

    id_producto = int(raw)
    cursor = conn.cursor()
    query = """
        SELECT *
        FROM rob.Producto
        WHERE Id_producto = ?;
    """
    cursor.execute(query, (id_producto,))
    row = cursor.fetchone()

    if row is None:
        print(f"No se encontró producto con Id_producto = {id_producto}")
    else:
        print("\nProducto encontrado:\n")
        imprimir_producto(row)


def ver_ventas(conn):
    raw = input(
        "Ingresa el año del que quieres consultar las ventas (ej: 2025): ").strip()
    if not raw.isdigit():
        print("El año debe ser numérico.")
        return

    anio = int(raw)
    cursor = conn.cursor()

    query = """
        SELECT
            v.Id_venta,
            v.Fecha,
            p.Nombre_producto,
            v.Unidades_vendidas,
            v.Cantidad_neta,
            v.Venta_sin_IVA
        FROM rob.Ventas v
        INNER JOIN rob.Producto p ON p.Id_producto = v.Id_producto
        WHERE YEAR(v.Fecha) = ?
        ORDER BY v.Fecha ASC, v.Id_venta ASC;
    """
    cursor.execute(query, (anio,))
    rows = cursor.fetchall()

    if not rows:
        print(f"No hay ventas registradas para el año {anio}.")
        return

    print(f"\nVentas del año {anio}:\n")
    for r in rows:
        print(
            f"Id_venta: {r.Id_venta} | Fecha: {r.Fecha} | "
            f"Producto: {r.Nombre_producto} | "
            f"Unid: {r.Unidades_vendidas} | Neta: {r.Cantidad_neta} | "
            f"Sin IVA: {r.Venta_sin_IVA}"
        )

    query_total = """
        SELECT SUM(v.Venta_sin_IVA) AS Total
        FROM rob.Ventas v
        WHERE YEAR(v.Fecha) = ?;
    """
    cursor.execute(query_total, (anio,))
    total = cursor.fetchone()[0]

    total = float(total) if total is not None else 0.0
    print("\n-------------------------------")
    print(f"TOTAL VENTA SIN IVA {anio}: ${total:,.2f}")
    print("-------------------------------")


def actualizar_producto(conn):
    print("\n¿Qué deseas actualizar?")
    print("1) Nombre")
    print("2) Marca")
    print("3) Categoría")
    print("4) Estado")

    opcion = input("Selecciona una opción: ").strip()

    campos = {
        "1": "Nombre_producto",
        "2": "Marca",
        "3": "Categoria",
        "4": "Estado"
    }

    if opcion not in campos:
        print("Opción inválida.")
        return

    id_producto = input("Ingresa el Id del producto: ").strip()
    if not id_producto.isdigit():
        print("El Id debe ser numérico.")
        return

    if opcion == "3":
        CATEGORIAS_VALIDAS = {
            "alimento caja": "Alimento Caja",
            "aseo caja": "Aseo Caja",
            "aseo bolsa": "Aseo Bolsa"
        }
        raw = input("Nueva categoría: ").strip().lower()
        if raw not in CATEGORIAS_VALIDAS:
            print("Categoría inválida.")
            return
        nuevo_valor = CATEGORIAS_VALIDAS[raw]

    elif opcion == "4":
        ESTADOS_VALIDOS = {
            "0": 0,
            "1": 1
        }
        raw = input("Nuevo estado: ").strip().lower()
        if raw not in ESTADOS_VALIDOS:
            print("Estado inválido.")
            return
        nuevo_valor = ESTADOS_VALIDOS[raw]
    else:
        nuevo_valor = input("Ingresa el nuevo valor: ").strip()

    cursor = conn.cursor()

    # UPDATE
    query_update = f"""
        UPDATE rob.Producto
        SET {campos[opcion]} = ?
        WHERE Id_producto = ?;
    """
    cursor.execute(query_update, (nuevo_valor, int(id_producto)))
    conn.commit()

    query_select = """
        SELECT *
        FROM rob.Producto
        WHERE Id_producto = ?;
    """
    cursor.execute(query_select, (int(id_producto),))
    row = cursor.fetchone()

    if row is None:
        print("No se encontró el producto.")
    else:
        print("\nProducto actualizado correctamente:")
        imprimir_producto(row)


def ingresar_producto(conn):
    raw_id = input("Ingresa el Id del producto: ").strip()
    try:
        id_producto = int(raw_id)
        if id_producto <= 0:
            raise ValueError
    except ValueError:
        print("Id de producto inválido. Debe ser un entero positivo.")
        return

    nombre = input("Ingresa el nombre del producto: ").strip()
    marca = input("Ingresa la marca del producto: ").strip()

    CATEGORIAS_VALIDAS = {
        "alimento caja": "Alimento Caja",
        "aseo caja": "Aseo Caja",
        "aseo bolsa": "Aseo Bolsa"}

    raw_categoria = input(
        "Ingresa la categoría (Alimento Caja / Aseo Caja / Aseo Bolsa): ").strip().lower()

    if raw_categoria not in CATEGORIAS_VALIDAS:
        print("Categoría inválida.")
        print("Opciones válidas:")
        for c in CATEGORIAS_VALIDAS.values():
            print(f" - {c}")
        return

    categoria = CATEGORIAS_VALIDAS[raw_categoria]

    if not nombre:
        print("Nombre no puede estar vacío.")
        return

    raw_estado = input(
        "Ingresa el estado del producto (0=Inactivo, 1=Activo): ").strip()
    if raw_estado not in ("0", "1"):
        print("Estado inválido, debe ser 0 o 1.")
        return
    estado = int(raw_estado)

    cursor = conn.cursor()

    # Valida que ese id no exista en la base de datos
    cursor.execute(
        "SELECT 1 FROM rob.Producto WHERE Id_producto = ?;", (id_producto,))
    if cursor.fetchone() is not None:
        print(f"Ya existe un producto con Id_producto = {id_producto}.")
        return

    try:
        query_insert = """
            INSERT INTO rob.Producto (Id_producto, Nombre_producto, Marca, Categoria, Estado)
            VALUES (?, ?, ?, ?, ?);
        """
        cursor.execute(query_insert, (id_producto,
                       nombre, marca, categoria, estado))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error insertando el producto: {e}")
        return

    cursor.execute(
        "SELECT * FROM rob.Producto WHERE Id_producto = ?;", (id_producto,))
    r = cursor.fetchone()

    if r is None:
        print("Se insertó, pero no se pudo leer el producto registrado.")
        return

    print("\nProducto registrado:\n")
    print(
        f"Id_producto: {r.Id_producto} | Nombre: {r.Nombre_producto} | "
        f"Marca: {r.Marca} | Categoria: {r.Categoria} | "
        f"Estado: {r.Estado}"
    )


def eliminar_producto(conn):
    raw_id = input("Ingresa el Id del producto a eliminar: ").strip()
    id_producto = int(raw_id)
    cursor = conn.cursor()
    # Valida que ese id exista en la base de datos
    cursor.execute(
        "SELECT 1 FROM rob.Producto WHERE Id_producto = ?;", (id_producto,))
    if cursor.fetchone() is None:
        print(f"No existe un producto con el Id_producto = {id_producto}.")
        return

    try:
        query_delete = """
            DELETE FROM rob.Producto 
            WHERE Id_producto = ?;
        """
        cursor.execute(query_delete, (id_producto,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error eliminando el producto: {e}")
        return


def ingresar_venta(conn):
    raw_id = input(
        "Ingrese el Id del producto del que registrará la venta: ").strip()
    id_producto = int(raw_id)
    cursor = conn.cursor()
    # Valida que ese id exista en la base de datos
    cursor.execute(
        "SELECT 1 FROM rob.Producto WHERE Id_producto = ?;", (id_producto,))
    if cursor.fetchone() is None:
        print(f"No existe un producto con el Id_producto = {id_producto}.")
        return

    raw_unid = input("Ingrese el número de unidades vendidas: ").strip()
    try:
        unidades = int(raw_unid)
    except ValueError:
        print("Unidades inválidas. Debe ser un entero.")
        return

    raw_neta = input("Ingrese la cantidad neta: ").strip()
    try:
        cantidad_neta = int(raw_neta)
    except ValueError:
        print("Cantidad neta inválida. Debe ser un entero.")
        return

    raw_venta = input(
        "Ingrese el valor de la venta sin IVA: ").strip().replace(",", ".")
    try:
        venta_sin_iva = float(raw_venta)
    except ValueError:
        print("Venta sin IVA inválida. Debe ser numérica (ej: 1234.56).")
        return

    raw_fecha = input("Ingrese la fecha en formato AAAA-MM-DD: ").strip()
    try:
        fecha = date.fromisoformat(raw_fecha)
    except ValueError:
        print("Fecha inválida. Usa el formato AAAA-MM-DD (ej: 2025-10-31).")
        return

    cursor = conn.cursor()

    try:
        query_insert = """
            INSERT INTO rob.Ventas (Id_producto, Unidades_vendidas, Cantidad_neta, Venta_sin_IVA, Fecha)
            VALUES (?, ?, ?, ?, ?);
        """
        cursor.execute(query_insert, (id_producto, unidades,
                       cantidad_neta, venta_sin_iva, fecha))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error insertando la venta: {e}")
        return

    query_select = """
        SELECT TOP 10
            v.Id_venta,
            v.Fecha,
            p.Nombre_producto,
            v.Unidades_vendidas,
            v.Cantidad_neta,
            v.Venta_sin_IVA
        FROM rob.Ventas v
        LEFT JOIN rob.Producto p ON p.Id_producto = v.Id_producto
        WHERE v.Id_producto = ?
        ORDER BY v.Fecha DESC, v.Id_venta DESC;
    """
    cursor.execute(query_select, (id_producto,))
    rows = cursor.fetchall()

    if not rows:
        print("Venta registrada, pero no se pudo listar historial (revisar consulta).")
        return

    print("\nVenta registrada. Últimas ventas de este producto:\n")
    for r in rows:
        print(
            f"Id_venta: {r.Id_venta} | Fecha: {r.Fecha} | "
            f"Producto: {r.Nombre_producto} | Unid: {r.Unidades_vendidas} | "
            f"Neta: {r.Cantidad_neta} | Sin IVA: {r.Venta_sin_IVA}"
        )


def ingresar_movimiento(conn):
    cursor = conn.cursor()
    raw_id = input(
        "Ingrese el Id del producto del que registrará un movimiento: ").strip()
    if not raw_id.isdigit():
        print("Id inválido. Debe ser numérico.")
        return

    id_producto = int(raw_id)

    cursor.execute(
        "SELECT Nombre_producto FROM rob.Producto WHERE Id_producto = ?;", (id_producto,))
    prod = cursor.fetchone()
    if prod is None:
        print(f"No existe un producto con el Id_producto = {id_producto}.")
        return
    nombre_producto = prod[0]

    TIPO_VALIDO = {
        "entrada": "Entrada",
        "salida": "Salida",
        "ajuste": "Ajuste",
        "traslado": "Traslado",
    }
    raw_tipo = input(
        "Ingrese el tipo de movimiento (Entrada/Salida/Ajuste/Traslado): ").strip().lower()
    if raw_tipo not in TIPO_VALIDO:
        print("Tipo inválido. Opciones válidas:")
        for v in TIPO_VALIDO.values():
            print(f" - {v}")
        return
    tipo = TIPO_VALIDO[raw_tipo]

    raw_cantidad = input("Ingrese la cantidad: ").strip()
    try:
        cantidad = int(raw_cantidad)
        if cantidad <= 0:
            print("La cantidad debe ser un entero positivo.")
            return
    except ValueError:
        print("Cantidad inválida. Debe ser un entero.")
        return

    raw_fecha = input("Ingrese la fecha en formato AAAA-MM-DD: ").strip()
    try:
        fecha = date.fromisoformat(raw_fecha)
    except ValueError:
        print("Fecha inválida. Usa el formato AAAA-MM-DD (ej: 2025-10-31).")
        return

    ORIGEN_VALIDO = {
        "compra": "Compra",
        "venta": "Venta",
        "devolucion": "Devolución",
        "devolución": "Devolución",
        "merma": "Merma",
    }
    raw_origen = input(
        "Ingrese el origen (Compra/Venta/Inventario físico/Devolución/Merma): ").strip().lower()
    if raw_origen not in ORIGEN_VALIDO:
        print("Origen inválido. Opciones válidas:")
        for v in set(ORIGEN_VALIDO.values()):
            print(f" - {v}")
        return
    origen = ORIGEN_VALIDO[raw_origen]

    try:
        query_insert = """
            INSERT INTO rob.MovimientosInventario (Id_producto, Tipo_movimiento, Cantidad, Fecha, Origen)
            VALUES (?, ?, ?, ?, ?);
        """
        cursor.execute(query_insert, (id_producto,
                       tipo, cantidad, fecha, origen))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error insertando el movimiento: {e}")
        return

    query_select = """
        SELECT TOP 10
            m.Id_movimiento,
            m.Fecha,
            p.Nombre_producto,
            m.Tipo_movimiento,
            m.Cantidad,
            m.Origen
        FROM rob.MovimientosInventario AS m
        INNER JOIN rob.Producto AS p ON p.Id_producto = m.Id_producto
        WHERE m.Id_producto = ?
        ORDER BY m.Fecha DESC, m.Id_movimiento DESC;
    """
    cursor.execute(query_select, (id_producto,))
    rows = cursor.fetchall()

    print(f"\nMovimiento registrado para: {nombre_producto}")
    print("Últimos 10 movimientos de este producto:\n")
    for r in rows:
        print(
            f"Id_mov: {r.Id_movimiento} | Fecha: {r.Fecha} | "
            f"Producto: {r.Nombre_producto} | "
            f"Tipo: {r.Tipo_movimiento} | Cant: {r.Cantidad} | Origen: {r.Origen}"
        )


def ver_movimientos(conn):
    cursor = conn.cursor()

    TIPO_VALIDO = {
        "entrada": "Entrada",
        "salida": "Salida",
    }
    ORIGEN_VALIDO = {
        "compra": "Compra",
        "venta": "Venta",
        "devolucion": "Devolución",
        "devolución": "Devolución",
        "merma": "Merma",
    }

    print("\n¿Qué movimientos deseas ver?")
    print("1) Por producto (Id_producto)")
    print("2) Por tipo (Entrada/Salida)")
    print("3) Por año")
    print("4) Por origen")
    print("5) Ver todos")
    opcion = input("Selecciona una opción: ").strip()

    where_clauses = []
    params = []

    if opcion == "1":
        raw_id = input("Ingresa el Id_producto: ").strip()
        if not raw_id.isdigit():
            print("Id inválido. Debe ser numérico.")
            return
        id_producto = int(raw_id)
        cursor.execute(
            "SELECT 1 FROM rob.Producto WHERE Id_producto = ?;", (id_producto,))
        if cursor.fetchone() is None:
            print(f"No existe un producto con Id_producto = {id_producto}.")
            return

        where_clauses.append("m.Id_producto = ?")
        params.append(id_producto)

    elif opcion == "2":
        raw_tipo = input(
            "Ingresa el tipo (Entrada/Salida/Ajuste/Traslado): ").strip().lower()
        if raw_tipo not in TIPO_VALIDO:
            print("Tipo inválido. Opciones válidas:")
            for v in TIPO_VALIDO.values():
                print(f" - {v}")
            return
        tipo = TIPO_VALIDO[raw_tipo]
        where_clauses.append("m.Tipo_movimiento = ?")
        params.append(tipo)

    elif opcion == "3":
        raw_anio = input("Ingresa el año (ej: 2025): ").strip()
        if not raw_anio.isdigit():
            print("Año inválido. Debe ser numérico.")
            return
        anio = int(raw_anio)
        where_clauses.append("YEAR(m.Fecha) = ?")
        params.append(anio)

    elif opcion == "4":
        raw_origen = input(
            "Ingresa el origen (Compra/Venta/Devolución/Merma): ").strip().lower()
        if raw_origen not in ORIGEN_VALIDO:
            print("Origen inválido. Opciones válidas:")
            for v in set(ORIGEN_VALIDO.values()):
                print(f" - {v}")
            return
        origen = ORIGEN_VALIDO[raw_origen]
        where_clauses.append("m.Origen = ?")
        params.append(origen)

    elif opcion == "5":
        pass
    else:
        print("Opción inválida.")
        return

    query = """
        SELECT TOP 200
            m.Id_movimiento,
            m.Fecha,
            p.Nombre_producto,
            m.Tipo_movimiento,
            m.Cantidad,
            m.Origen
        FROM rob.MovimientosInventario AS m
        INNER JOIN rob.Producto AS p ON p.Id_producto = m.Id_producto
    """

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY m.Fecha DESC, m.Id_movimiento DESC;"

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()

    if not rows:
        print("\nNo se encontraron movimientos con ese criterio.")
        return

    print(f"\nMostrando {len(rows)} movimientos:\n")
    for r in rows:
        print(
            f"Id_mov: {r.Id_movimiento} | Fecha: {r.Fecha} | "
            f"Producto: {r.Nombre_producto} | "
            f"Tipo: {r.Tipo_movimiento} | Cant: {r.Cantidad} | Origen: {r.Origen}"
        )


def ver_inventario(conn):
    cursor = conn.cursor()

    print("\n¿Cómo desea ver el inventario?")
    print("1) Por producto (Id_producto)")
    print("2) Por mes y año")
    print("3) Por año")
    print("4) Ver todos")
    opcion = input("Selecciona una opción: ").strip()

    where_clauses = []
    params = []

    if opcion == "1":
        raw_id = input("Ingresa el Id_producto: ").strip()
        if not raw_id.isdigit():
            print("Id inválido. Debe ser numérico.")
            return

        id_producto = int(raw_id)

        cursor.execute(
            "SELECT 1 FROM rob.Producto WHERE Id_producto = ?;", (id_producto,))
        if cursor.fetchone() is None:
            print(f"No existe un producto con Id_producto = {id_producto}.")
            return

        where_clauses.append("c.Id_producto = ?")
        params.append(id_producto)

    elif opcion == "2":
        raw_month = input("Ingresa el mes (1-12, ej: 1 o 01): ").strip()
        if not raw_month.isdigit():
            print("Mes inválido. Debe ser numérico.")
            return

        month = int(raw_month)
        if month < 1 or month > 12:
            print("Mes inválido. Debe estar entre 1 y 12.")
            return

        where_clauses.append("MONTH(c.Fecha) = ?")
        params.append(month)

        raw_anio = input(
            "Ingresa el año (ej: 2025) para filtrar el mes: ").strip()
        if not raw_anio.isdigit():
            print("Año inválido. Debe ser numérico.")
            return

        anio = int(raw_anio)
        where_clauses.append("YEAR(c.Fecha) = ?")
        params.append(anio)

    elif opcion == "3":
        raw_anio = input("Ingresa el año (ej: 2025): ").strip()
        if not raw_anio.isdigit():
            print("Año inválido. Debe ser numérico.")
            return

        anio = int(raw_anio)
        where_clauses.append("YEAR(c.Fecha) = ?")
        params.append(anio)

    elif opcion == "4":
        pass
    else:
        print("Opción inválida.")
        return

    query = """
        SELECT
            c.Fecha,
            p.Nombre_producto,
            c.Cantidad
        FROM rob.Inventario AS c
        INNER JOIN rob.Producto AS p ON p.Id_producto = c.Id_producto
    """

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY c.Fecha DESC, c.Id_inventario DESC;"

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()

    if not rows:
        print("\nNo se encontraron inventarios con ese criterio.")
        return

    print(f"\nMostrando {len(rows)} inventario:\n")
    for r in rows:
        print(
            f"Fecha: {r.Fecha} | "
            f"Producto: {r.Nombre_producto} | "
            f"Cant_contada: {r.Cantidad}"
        )


def ingresar_conteo(conn):
    cursor = conn.cursor()

    raw_id = input("Ingresa el Id_producto a contar: ").strip()
    if not raw_id.isdigit():
        print("Id inválido. Debe ser numérico.")
        return
    id_producto = int(raw_id)

    cursor.execute("""
        SELECT Nombre_producto
        FROM rob.Producto
        WHERE Id_producto = ?;
    """, (id_producto,))
    prod = cursor.fetchone()
    if prod is None:
        print(f"No existe un producto con Id_producto = {id_producto}.")
        return

    raw_cant = input("Ingresa la cantidad contada: ").strip()
    try:
        cantidad_contada = int(raw_cant)
        if cantidad_contada < 0:
            raise ValueError
    except ValueError:
        print("Cantidad inválida. Debe ser un entero >= 0.")
        return

    query_insert = """
        INSERT INTO rob.Conteo (Id_producto, Cantidad_contada, Fecha)
        VALUES (?, ?, SYSDATETIME());
    """

    try:
        cursor.execute(query_insert, (id_producto, cantidad_contada))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error insertando el conteo: {e}")
        return

    query_select = """
        SELECT TOP 30
            c.Fecha,
            p.Nombre_producto,
            c.Cantidad_contada
        FROM rob.Conteo AS c
        INNER JOIN rob.Producto AS p ON p.Id_producto = c.Id_producto
        WHERE c.Id_producto = ?;
    """
    cursor.execute(query_select, (id_producto,))
    row = cursor.fetchone()

    if row is None:
        print("Conteo insertado, pero no se pudo leer con JOIN.")
        return

    print("\nConteo registrado:\n")
    print(
        f"Fecha: {row.Fecha} | "
        f"Producto: {row.Nombre_producto} | "
        f"Cantidad_contada: {row.Cantidad_contada}"
    )


def menu_consultas(conn):
    while True:
        print("\n==============================")
        print("          CONSULTAS           ")
        print("==============================")
        print("1) Buscar productos por categoría")
        print("2) Buscar productos por nombre")
        print("3) Buscar productos por Id")
        print("4) Ver ventas por año")
        print("5) Ver movimientos")
        print("6) Ver inventario")
        print("0) Volver")
        op = input("Selecciona una opción: ").strip()

        if op == "1":
            buscar_por_categoria(conn)
            pause()
        elif op == "2":
            buscar_por_nombre(conn)
            pause()
        elif op == "3":
            buscar_por_id(conn)
            pause()
        elif op == "4":
            ver_ventas(conn)
            pause()
        elif op == "5":
            ver_movimientos(conn)
            pause()
        elif op == "6":
            ver_inventario(conn)
            pause()
        elif op == "0":
            return
        else:
            print("Opción inválida.")


def menu_gestion(conn):
    while True:
        print("\n==============================")
        print("       GESTIÓN DE DATOS       ")
        print("==============================")
        print("1) Ingresar producto")
        print("2) Actualizar producto")
        print("3) Eliminar producto")
        print("4) Registrar venta")
        print("5) Registrar movimiento inventario")
        print("6) Registrar conteo de inventario")
        print("0) Volver")
        op = input("Selecciona una opción: ").strip()

        if op == "1":
            ingresar_producto(conn)
            pause()
        elif op == "2":
            actualizar_producto(conn)
            pause()
        elif op == "3":
            eliminar_producto(conn)
            pause()
        elif op == "4":
            ingresar_venta(conn)
            pause()
        elif op == "5":
            ingresar_movimiento(conn)
            pause()
        elif op == "6":
            ingresar_conteo(conn)
            pause()
        elif op == "0":
            return
        else:
            print("Opción inválida.")


# =========================
# Menú principal
# =========================

def main():
    try:
        conn = pyodbc.connect(CONN_STR)
    except Exception as e:
        print("Error conectando a la Base de datos")
        print(e)
        return

    print("Conectado.\n")

    try:
        while True:
            print("\n==============================")
            print("        MENÚ PRINCIPAL        ")
            print("==============================")
            print("1) Consultas")
            print("2) Gestión de datos")
            print("0) Salir")
            opcion = input("Selecciona una opción: ").strip()

            if opcion == "1":
                menu_consultas(conn)
            elif opcion == "2":
                menu_gestion(conn)
            elif opcion == "0":
                print("Saliendo...")
                break
            else:
                print("Opción inválida. Intenta de nuevo.")

    finally:
        conn.close()
        print("Conexión cerrada.")


if __name__ == "__main__":
    main()
