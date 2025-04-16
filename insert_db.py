from database import connect_db

def insert_patient(values):
    conn = None
    cursor = None
    try:
        conn = connect_db()
        cursor = conn.cursor()

        query = """
        INSERT INTO patients (
            Epigastralgie,
            Metastases_Hepatiques,
            Dénutrition,
            tabac,
            Mucineux,
            Ulcéro_Bourgeonnant,
            Adenopaties,
            Ulcere_gastrique,
            infiltrant,
            Cardiopathie,
            Sténosant,
            DECES
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(query, values)
        conn.commit()

    except Exception as e:
        print(f"❌ Erreur base de données : {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
