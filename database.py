import mysql.connector

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",          # pas de mot de passe
        database="ccc"
    )
