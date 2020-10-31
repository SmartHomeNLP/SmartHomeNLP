import mysql.connector
database_password = "BestPass0701"
mydb = mysql.connector.connect(user="root",
                                   password=database_password,
                                   host="localhost")

mycursor = mydb.cursor()