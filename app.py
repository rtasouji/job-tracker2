import streamlit as st
import requests
import pandas as pd
import tldextract
import psycopg2
import json
from collections import defaultdict
import datetime
import os
import plotly.graph_objects as go
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("job-tracker")

# Database connection function
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("❌ ERROR: DB_URL environment variable is not set!")

def get_db_connection():
    return psycopg2.connect(DB_URL, sslmode="require")

# Initialize database tables (including campaigns table with keywords)
def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create or update share_of_voice table (existing)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS share_of_voice (
            id SERIAL PRIMARY KEY,
            domain TEXT NOT NULL,
            sov FLOAT NOT NULL,
            appearances INT DEFAULT 0,
            avg_v_rank FLOAT DEFAULT 0,
            avg_h_rank FLOAT DEFAULT 0,
            date DATE NOT NULL,
            campaign_id TEXT NOT NULL
        );
    """)

    # Create campaigns table to store campaign details and keywords
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            job_titles TEXT NOT NULL,  -- JSON or string of job titles
            locations TEXT NOT NULL,   -- JSON or string of locations
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("ALTER TABLE share_of_voice ADD COLUMN IF NOT EXISTS campaign_id TEXT NOT NULL DEFAULT 'default';")
    conn.commit()
    cursor.close()
    conn.close()

initialize_database()

# Load jobs (keywords and locations) from the database
def load_jobs(campaign_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT job_titles, locations FROM campaigns WHERE campaign_id = %s", (campaign_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        st.error(f"⚠️ Campaign '{campaign_id}' not found!")
        return []

    job_titles_str, locations_str = result
    try:
        job_titles = json.loads(job_titles_str) if job_titles_str else []
        locations = json.loads(locations_str) if locations_str else []
    except json.JSONDecodeError:
        # Fallback: assume comma-separated strings if JSON fails
        job_titles = [t.strip() for t in job_titles_str.split(',')] if job_titles_str else []
        locations = [l.strip() for l in locations_str.split(',')] if locations_str else []

    if len(job_titles) != len(locations):
        st.error(f"⚠️ Mismatch between job titles and locations for campaign '{campaign_id}'!")
        return []

    return [{"job_title": title, "location": loc} for title, loc in zip(job_titles, locations)]

# Fetch Google Jobs Results from SerpAPI
def get_google_jobs_results(query, location):
    SERP_API_KEY = os.getenv("SERP_API_KEY")
    if not SERP_API_KEY:
        raise ValueError("❌ ERROR: SERP_API_KEY environment variable is not set!")

    logger.info(f"Fetching results for query: {query} in location: {location} with API key: {SERP_API_KEY[:4]}...{SERP_API_KEY[-4:]}")
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "hl": "en",
        "api_key": SERP_API_KEY
    }
    
    response = requests.get(url, params=params)
    logger.info(f"SerpAPI response status code: {response.status_code}")
    if response.status_code != 200:
        logger.error(f"Failed to fetch data from SerpAPI. Status Code: {response.status_code}, Response: {response.text}")
        raise RuntimeError(f"❌ ERROR: Failed to fetch data from SerpAPI. Status Code: {response.status_code}")
    
    results = response.json().get("jobs_results", [])
    logger.info(f"Received {len(results)} job results for query: {query} in {location}")
    return results

# Compute Share of Voice & Additional Metrics for a specific campaign
def compute_sov(campaign_id):
    logger.info(f"Starting compute_sov for campaign {campaign_id}")
    domain_sov = defaultdict(float)
    domain_appearances = defaultdict(int)
    domain_v_rank = defaultdict(list)
    domain_h_rank = defaultdict(list)

    jobs_data = load_jobs(campaign_id)
    logger.info(f"Loaded {len(jobs_data)} job queries from database for campaign {campaign_id}")
    total_sov = 0  

    for job_query in jobs_data:
        job_title = job_query["job_title"]
        location = job_query["location"]
        logger.info(f"Processing job query: {job_title} in {location} for campaign {campaign_id}")

        try:
            jobs = get_google_jobs_results(job_title, location)
            logger.info(f"Retrieved {len(jobs)} job results for query in campaign {campaign_id}")
            if not jobs:
                logger.warning(f"No jobs found for query: {job_title} in {location} for campaign {campaign_id}")
            
            for job_rank, job in enumerate(jobs, start=1):
                apply_options = job.get("apply_options", [])
                V = 1 / job_rank  
                logger.debug(f"Processing job at rank {job_rank} with apply_options: {apply_options}")

                for link_order, option in enumerate(apply_options, start=1):
                    if "link" in option:
                        domain = extract_domain(option["link"])
                        H = 1 / link_order  
                        weight = V * H  
                        logger.debug(f"Processing domain {domain} with weight {weight}")
                        domain_sov[domain] += weight  
                        domain_appearances[domain] += 1
                        domain_v_rank[domain].append(job_rank)
                        domain_h_rank[domain].append(link_order)
                        total_sov += weight  
        except Exception as e:
            logger.error(f"Error processing job query {job_title} for campaign {campaign_id}: {str(e)}")
            continue

    if total_sov > 0:
        logger.info(f"Computed SoV for {len(domain_sov)} domains with total SoV: {total_sov} for campaign {campaign_id}")
        domain_sov = {domain: round((sov / total_sov) * 100, 4) for domain, sov in domain_sov.items()}
    else:
        logger.warning(f"No SoV computed for campaign {campaign_id} due to zero total SoV")
    
    domain_avg_v_rank = {domain: round(sum(vr) / len(vr), 2) for domain, vr in domain_v_rank.items() if vr}
    domain_avg_h_rank = {domain: round(sum(hr) / len(hr), 2) for domain, hr in domain_h_rank.items() if hr}

    logger.info(f"Returning SoV data for {len(domain_sov)} domains for campaign {campaign_id}")
    return domain_sov, domain_appearances, domain_avg_v_rank, domain_avg_h_rank

# Extract Domain from URL
def extract_domain(url):
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
    return domain.lower().replace("www.", "")

# Save data to database with campaign_id
def save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, campaign_id):
    logger.info(f"Attempting to save data for campaign {campaign_id}")
    if not sov_data:
        logger.warning(f"No SoV data to save to database for campaign {campaign_id}. Check if job results were retrieved.")
        return
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        today = datetime.date.today()
        logger.info(f"Saving data for date: {today} for campaign {campaign_id}")

        for domain in sov_data:
            logger.info(f"Inserting data for domain: {domain}, SoV: {sov_data[domain]}, Appearances: {appearances[domain]}, "
                        f"Avg V Rank: {avg_v_rank[domain]}, Avg H Rank: {avg_h_rank[domain]} for campaign {campaign_id}")
            cursor.execute("""
                INSERT INTO share_of_voice (domain, sov, appearances, avg_v_rank, avg_h_rank, date, campaign_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (domain, round(sov_data[domain], 2), appearances[domain], 
                  avg_v_rank[domain], avg_h_rank[domain], today, campaign_id))
        conn.commit()
        logger.info(f"Database commit successful for {len(sov_data)} domains in campaign {campaign_id}")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database error for campaign {campaign_id}: {str(e)}")
        raise

# Retrieve historical data for a specific campaign
def get_historical_data(start_date, end_date, campaign_id):
    logger.info(f"Retrieving historical data for campaign {campaign_id} from {start_date} to {end_date}")
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'share_of_voice'
        );
    """)
    
    table_exists = cursor.fetchone()[0]
    
    if not table_exists:
        logger.warning("Share_of_voice table does not exist.")
        st.warning("⚠️ No data available yet.")
        cursor.close()
        conn.close()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    query = """
        SELECT domain, date, sov, appearances, avg_v_rank, avg_h_rank
        FROM share_of_voice 
        WHERE date BETWEEN %s AND %s AND campaign_id = %s
    """
    logger.info(f"Executing query for campaign {campaign_id} with date range {start_date} to {end_date}")
    cursor.execute(query, (start_date, end_date, campaign_id))
    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} rows for campaign {campaign_id}")

    df = pd.DataFrame(rows, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank"])

    cursor.close()
    conn.close()

    if df.empty:
        logger.warning(f"No data available for campaign {campaign_id} in the selected date range.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date  
    df_agg = df.groupby(["domain", "date"], as_index=False).agg({
        "sov": "mean",
        "appearances": "sum",
        "avg_v_rank": "mean",
        "avg_h_rank": "mean"
    })

    df_sov = df_agg.pivot(index="domain", columns="date", values="sov").fillna(0)
    df_metrics = df_agg.pivot(index="domain", columns="date", values=["appearances", "avg_v_rank", "avg_h_rank"]).fillna(0)
    df_metrics = df_metrics.swaplevel(axis=1).sort_index(axis=1)

    if not df_sov.empty:
        most_recent_date = df_sov.columns[-1]  
        df_sov = df_sov.sort_values(by=most_recent_date, ascending=False)

    df_appearances = df_agg.pivot(index="domain", columns="date", values="appearances").fillna(0)

    logger.info(f"Returning data for {len(df_sov)} domains for campaign {campaign_id}")
    return df_sov, df_metrics, df_appearances

# New function to get total data across all campaigns
def get_total_historical_data(start_date, end_date):
    logger.info(f"Retrieving total historical data from {start_date} to {end_date}")
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'share_of_voice'
        );
    """)
    
    table_exists = cursor.fetchone()[0]
    
    if not table_exists:
        logger.warning("Share_of_voice table does not exist.")
        st.warning("⚠️ No data available yet.")
        cursor.close()
        conn.close()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    query = """
        SELECT domain, date, sov, appearances, avg_v_rank, avg_h_rank
        FROM share_of_voice 
        WHERE date BETWEEN %s AND %s
    """
    logger.info(f"Executing total query with date range {start_date} to {end_date}")
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} total rows across all campaigns")

    df = pd.DataFrame(rows, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank"])

    cursor.close()
    conn.close()

    if df.empty:
        logger.warning("No total data available for the selected date range.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date  
    df_agg = df.groupby(["domain", "date"], as_index=False).agg({
        "sov": "sum",  # Sum SoV across campaigns for the same domain and date
        "appearances": "sum",  # Sum appearances across campaigns
        "avg_v_rank": "mean",  # Average vertical rank across campaigns
        "avg_h_rank": "mean"   # Average horizontal rank across campaigns
    })

    df_sov = df_agg.pivot(index="domain", columns="date", values="sov").fillna(0)
    df_metrics = df_agg.pivot(index="domain", columns="date", values=["appearances", "avg_v_rank", "avg_h_rank"]).fillna(0)
    df_metrics = df_metrics.swaplevel(axis=1).sort_index(axis=1)

    if not df_sov.empty:
        most_recent_date = df_sov.columns[-1]  
        df_sov = df_sov.sort_values(by=most_recent_date, ascending=False)

    df_appearances = df_agg.pivot(index="domain", columns="date", values="appearances").fillna(0)

    logger.info(f"Returning total data for {len(df_sov)} domains")
    return df_sov, df_metrics, df_appearances

# New function to fetch and store total data
def compute_and_store_total_data():
    logger.info("Computing and storing total data across all campaigns")
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT domain, date, sov, appearances, avg_v_rank, avg_h_rank, campaign_id
        FROM share_of_voice
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        logger.warning("No data available to compute total across campaigns")
        return

    # Aggregate all data across campaigns
    domain_sov = defaultdict(float)
    domain_appearances = defaultdict(int)
    domain_v_rank = defaultdict(list)
    domain_h_rank = defaultdict(list)

    for row in rows:
        domain, date, sov, appearances, avg_v_rank, avg_h_rank, _ = row
        domain_sov[domain] += sov
        domain_appearances[domain] += appearances
        domain_v_rank[domain].append(avg_v_rank)
        domain_h_rank[domain].append(avg_h_rank)

    # Calculate averages for ranks
    total_avg_v_rank = {domain: round(sum(ranks) / len(ranks), 2) for domain, ranks in domain_v_rank.items() if ranks}
    total_avg_h_rank = {domain: round(sum(ranks) / len(ranks), 2) for domain, ranks in domain_h_rank.items() if ranks}

    # Normalize SoV to 100% total (optional, depending on your needs)
    total_sov = sum(domain_sov.values())
    if total_sov > 0:
        domain_sov = {domain: round((sov / total_sov) * 100, 4) for domain, sov in domain_sov.items()}

    # Store total data with a special campaign_id (e.g., 'total')
    save_to_db(domain_sov, domain_appearances, total_avg_v_rank, total_avg_h_rank, "total")

# Create or update a campaign (store keywords in database)
def create_or_update_campaign(campaign_id, campaign_name, job_titles, locations):
    if not job_titles or not locations:
        st.error("⚠️ Please provide at least one job title and location!")
        return False

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Convert lists to JSON strings for storage
        job_titles_json = json.dumps(job_titles)
        locations_json = json.dumps(locations)
        cursor.execute("""
            INSERT INTO campaigns (campaign_id, name, job_titles, locations)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (campaign_id) DO UPDATE 
            SET name = EXCLUDED.name, job_titles = EXCLUDED.job_titles, locations = EXCLUDED.locations
        """, (campaign_id, campaign_name, job_titles_json, locations_json))
        conn.commit()
        logger.info(f"Campaign '{campaign_id}' created/updated successfully as '{campaign_name}' with {len(job_titles)} job titles and locations")
    except Exception as e:
        logger.error(f"Database error for campaign {campaign_id}: {str(e)}")
        conn.rollback()
        st.error(f"Database error: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()
    return True

# Delete a campaign and optionally its associated data
def delete_campaign(campaign_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Delete associated data from share_of_voice (optional, can be skipped for historical tracking)
        cursor.execute("DELETE FROM share_of_voice WHERE campaign_id = %s", (campaign_id,))
        logger.info(f"Deleted {cursor.rowcount} records from share_of_voice for campaign {campaign_id}")

        # Delete the campaign from campaigns table
        cursor.execute("DELETE FROM campaigns WHERE campaign_id = %s", (campaign_id,))
        if cursor.rowcount > 0:
            conn.commit()
            logger.info(f"Campaign '{campaign_id}' deleted successfully")
            st.success(f"Campaign '{campaign_id}' deleted successfully!")
        else:
            logger.warning(f"No campaign found with ID '{campaign_id}'")
            st.warning(f"No campaign found with ID '{campaign_id}'")
    except Exception as e:
        logger.error(f"Error deleting campaign {campaign_id}: {str(e)}")
        conn.rollback()
        st.error(f"Error deleting campaign: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# Streamlit UI
st.title("Google for Jobs Visibility Tracker")

# Navigation
page = st.sidebar.selectbox("Navigate", ["Visibility Tracker", "Campaign Management"])

if page == "Visibility Tracker":
    # Display Logo
    st.image("logo.png", width=200)

    # Date Range Selector with dynamic default last 30 days
    st.sidebar.header("Date Range Selector")
    today = datetime.date.today()  # Use current date dynamically
    default_start_date = today - datetime.timedelta(days=30)  # 30 days before today
    start_date = st.sidebar.date_input("Start Date", value=default_start_date)
    end_date = st.sidebar.date_input("End Date", value=today)

    # Ensure end_date is not before start_date
    if start_date > end_date:
        st.sidebar.error("End date must be after start date!")
        st.stop()

    # Campaign selector for visibility tracker (using campaign names)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT campaign_id, name FROM campaigns")
    campaigns = cursor.fetchall()
    cursor.close()
    conn.close()

    # Create a list of campaign names, including "Total" for all campaigns
    campaign_name_options = ["Total"]  # Total campaign label
    campaign_id_map = {"Total": "tota
