import psycopg2
import os
import datetime

def check_db(campaign_id="default"):
    try:
        conn = psycopg2.connect(os.getenv("DB_URL"), sslmode="require")
        cursor = conn.cursor()
        today = datetime.date.today()
        
        cursor.execute("SELECT COUNT(*) FROM share_of_voice WHERE date = %s AND campaign_id = %s", (today, campaign_id))
        count = cursor.fetchone()[0]
        
        print(f"Records found for today for campaign '{campaign_id}': {count}")
        
        if count == 0:
            print(f"⚠️ WARNING: No records were stored for today for campaign '{campaign_id}'!")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error checking database for campaign '{campaign_id}': {str(e)}")

if __name__ == "__main__":
    import sys
    campaign_id = "default" if len(sys.argv) < 2 else sys.argv[1]
    check_db(campaign_id)