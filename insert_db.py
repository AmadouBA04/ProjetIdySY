from database import connect_db

def insert_patient(values):
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
        raise e

    finally:
        cursor.close()
        conn.close()
