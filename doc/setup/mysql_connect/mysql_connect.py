
import mysql.connector

cnx = mysql.connector.connect(user='adminCDCU', password='schmee#schm44',
                              host='ndsw-cdcu-yello',
                              database='dbCDCUtest')

cursor = cnx.cursor()

query = ('SELECT name from dumbTable')

cursor.execute(query)

for (name,) in cursor:
  print('Name is ' + name)

cursor.close()
cnx.close()

