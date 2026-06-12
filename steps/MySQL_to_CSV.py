from dotenv import load_dotenv

load_dotenv()


# Step 1: Convert MYSQL Tables to CSVs (Databricks free doesn't do connector)
import mysql.connector
import csv
import os

#making a class so easy to use self.cursor and self.conn

class migration():
    def __init__(self) -> None:
        self.output_dir = os.environ.get("OUTPUT_DIR")
        self.connect()

    # connect to mysql and create cursor and connection objects to use
    def connect(
        self,
        user=os.environ.get("MYSQL_USER"),
        password=os.environ.get("MYSQL_PASSWORD"),
        host=os.environ.get("MYSQL_HOST"),
        database=os.environ.get("MYSQL_DATABASE_NAME"),
    ):
        # ssl disabled since local machine
        self.conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )

        self.conn.autocommit = True
        self.cursor = self.conn.cursor(buffered=True)

    #make CSV output directory
    def make_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    #get all table names
    def get_table_names(self):
        self.cursor.execute("SHOW TABLES")
        tables = [row[0] for row in self.cursor.fetchall()]

        return tables
    
    #select all rows from table and write to CSV
    def write_to_csv(self):
        tables = self.get_table_names()

        for table in tables:
            self.cursor.execute(f"SELECT * FROM {table}")
            rows = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]

            filepath= os.path.join(self.output_dir, f"{table}.csv")

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(column_names)
                writer.writerows(rows)
            
            print(f"exported {table} ({len(rows)} rows) -> {filepath}")

if __name__ == "__main__":
    new_migration = migration()

    new_migration.make_output_dir()
    new_migration.write_to_csv()