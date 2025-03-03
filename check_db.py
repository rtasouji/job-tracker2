import logging
import psycopg2
import os
import datetime
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("job-tracker")

def check_db(campaign_name="Default"):
    try:
        conn = psycopg2.connect(os.getenv("DB_URL"), sslmode="require")
        cursor = conn.cursor()
        today = datetime.date.today()
        
        logger.info(f"Checking database for campaign {campaign_name} on date {today}")
        cursor.execute("SELECT COUNT(*) FROM share_of_voice WHERE date = %s AND campaign_name = %s", (today, campaign_name))
        count = cursor.fetchone()[0]
        
        logger.info(f"Records found for today for campaign '{campaign_name}': {count}")
        if count == 0:
            logger.warning(f"⚠️ WARNING: No records were stored for today for campaign '{campaign_name}'!")
        
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error checking database for campaign '{campaign_name}': {str(e)}")

if __name__ == "__main__":
    campaign_name = "Default" if len(sys.argv) < 2 else sys.argv[1]
    check_db(campaign_name)
