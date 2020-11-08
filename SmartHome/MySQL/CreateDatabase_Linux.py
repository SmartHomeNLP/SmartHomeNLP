import mysql.connector
import mariadb

# Get the passowrd
with open('../password.txt', 'r') as file:
    database_password = file.readline().strip()

# Create a connection

mydb = mariadb.connect(
        user="db_user",
        password="db_user_passwd",
        host="192.0.2.1",
        port=3306
    )

print(mydb)

# Create a database named 'reddit_smarthome'
mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE reddit_smarthome")

# Check if it has been created
mycursor.execute("SHOW DATABASES")

for x in mycursor:
    print(x)