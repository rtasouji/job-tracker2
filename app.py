import streamlit as st
import requests
import pandas as pd
import tldextract
import psycopg2
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

# Initialize database tables (including campaigns table)
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
            campaign_id TEXT NOT NULL  -- Add campaign_id to link data to campaigns
        );
    """)

    # Create campaigns table to store campaign details
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            csv_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("ALTER TABLE share_of_voice ADD COLUMN IF NOT EXISTS campaign_id TEXT NOT NULL DEFAULT 'default';")
    conn.commit()
    cursor.close()
    conn.close()

initialize_database()

# Load jobs from a specific campaign's CSV
def load_jobs(campaign_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT csv_path FROM campaigns WHERE campaign_id = %s", (campaign_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        st.error(f"⚠️ Campaign '{campaign_id}' not found!")
        return []

    csv_path = result[0]
    if not os.path.exists(csv_path):
        st.error(f"⚠️ CSV file '{csv_path}' not found for campaign '{campaign_id}'!")
        return []

    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

# Fetch Google Jobs Results from SerpAPI (unchanged)
def get_google_jobs_results(query, location):
    SERP_API_KEY = os.getenv("SERP_API_KEY")
    if not SERP_API_KEY:
        raise ValueError("❌ ERROR: SERP_API_KEY environment variable is not set!")

    url = "https://serpapi.com/search"
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "hl": "en",
        "api_key": SERP_API_KEY
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"❌ ERROR: Failed to fetch data from SerpAPI. Status Code: {response.status_code}")
    return response.json().get("jobs_results", [])

# Compute Share of Voice & Additional Metrics for a specific campaign
def compute_sov(campaign_id):
    domain_sov = defaultdict(float)
    domain_appearances = defaultdict(int)
    domain_v_rank = defaultdict(list)
    domain_h_rank = defaultdict(list)

    jobs_data = load_jobs(campaign_id)
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

                for link_order, option in enumerate(apply_options, start=1):
                    if "link" in option:
                        domain = extract_domain(option["link"])
                        H = 1 / link_order  
                        weight = V * H  
                        domain_sov[domain] += weight  
                        domain_appearances[domain] += 1
                        domain_v_rank[domain].append(job_rank)
                        domain_h_rank[domain].append(link_order)
                        total_sov += weight  

        except Exception as e:
            logger.error(f"Error processing job query {job_title} for campaign {campaign_id}: {str(e)}")
            continue

    if total_sov > 0:
        domain_sov = {domain: round((sov / total_sov) * 100, 4) for domain, sov in domain_sov.items()}
    
    domain_avg_v_rank = {domain: round(sum(vr) / len(vr), 2) for domain, vr in domain_v_rank.items() if vr}
    domain_avg_h_rank = {domain: round(sum(hr) / len(hr), 2) for domain, hr in domain_h_rank.items() if hr}

    return domain_sov, domain_appearances, domain_avg_v_rank, domain_avg_h_rank

# Extract Domain from URL (unchanged)
def extract_domain(url):
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
    return domain.lower().replace("www.", "")

# Save data to database with campaign_id
def save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, campaign_id):
    logger.info(f"Saving data for {len(sov_data)} domains to database for campaign {campaign_id}")
    
    if not sov_data:
        logger.warning(f"No SoV data to save to database for campaign {campaign_id}")
        return
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        today = datetime.date.today()
        logger.info(f"Saving data for date: {today} for campaign {campaign_id}")

        for domain in sov_data:
            logger.info(f"Inserting data for domain: {domain}, SoV: {sov_data[domain]} for campaign {campaign_id}")
            cursor.execute("""
                INSERT INTO share_of_voice (domain, sov, appearances, avg_v_rank, avg_h_rank, date, campaign_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (domain, round(sov_data[domain], 2), appearances[domain], 
                  avg_v_rank[domain], avg_h_rank[domain], today, campaign_id))

        conn.commit()
        logger.info("Database commit successful for campaign {campaign_id}")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database error for campaign {campaign_id}: {str(e)}")
        raise

# Retrieve historical data for a specific campaign
def get_historical_data(start_date, end_date, campaign_id):
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
        st.warning("⚠️ No data available yet.")
        cursor.close()
        conn.close()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    query = """
        SELECT domain, date, sov, appearances, avg_v_rank, avg_h_rank
        FROM share_of_voice 
        WHERE date BETWEEN %s AND %s AND campaign_id = %s
    """
    cursor.execute(query, (start_date, end_date, campaign_id))
    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank"])

    cursor.close()
    conn.close()

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

    return df_sov, df_metrics, df_appearances

# Create or update a campaign
def create_or_update_campaign(campaign_id, campaign_name, csv_file):
    if csv_file is not None:
        csv_path = f"campaigns/{campaign_id}_jobs.csv"
        os.makedirs("campaigns", exist_ok=True)
        # Save the uploaded file to disk
        with open(csv_path, "wb") as f:
            f.write(csv_file.getbuffer())
    else:
        st.error("⚠️ Please upload a CSV file for the campaign!")
        return False

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO campaigns (campaign_id, name, csv_path)
        VALUES (%s, %s, %s)
        ON CONFLICT (campaign_id) DO UPDATE 
        SET name = EXCLUDED.name, csv_path = EXCLUDED.csv_path
    """, (campaign_id, campaign_name, csv_path))
    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"Campaign '{campaign_id}' created/updated successfully with CSV at {csv_path}")
    return True

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

    # Campaign selector for visibility tracker
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT campaign_id, name FROM campaigns")
    campaigns = cursor.fetchall()
    cursor.close()
    conn.close()

    campaign_options = ["default"] + [c[0] for c in campaigns]
    selected_campaign = st.sidebar.selectbox("Select Campaign", campaign_options, index=0)

    # Fetch & Store Data for the selected campaign
    if st.button("Fetch & Store Data"):
        sov_data, appearances, avg_v_rank, avg_h_rank = compute_sov(selected_campaign)
        save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, selected_campaign)
        st.success(f"Data stored successfully for campaign '{selected_campaign}'!")

    # Show Historical Trends for the selected campaign
    st.write("### Visibility Over Time")
    df_sov, df_metrics, df_appearances = get_historical_data(start_date, end_date, selected_campaign)

    if not df_sov.empty:
        # Share of Voice Chart
        top_domains = df_sov.iloc[:15]
        fig1 = go.Figure()
        for domain in top_domains.index:
            fig1.add_trace(go.Scatter(
                x=top_domains.columns, 
                y=top_domains.loc[domain], 
                mode="markers+lines", 
                name=domain
            ))
        fig1.update_layout(
            title=f"Domains Visibility Over Time for Campaign '{selected_campaign}'",
            xaxis_title="Date",
            yaxis_title="Share of Voice (%)",
            updatemenus=[{"buttons": [{"args": [{"visible": True}], "label": "Show All", "method": "update"},
                                     {"args": [{"visible": "legendonly"}], "label": "Hide All", "method": "update"}],
                          "direction": "right", "showactive": True, "x": 1, "xanchor": "right", "y": 1.15, "yanchor": "top"}]
        )
        st.plotly_chart(fig1)
        st.write("#### Table of Visibility Score Data")
        st.dataframe(df_sov.style.format("{:.2f}"))

        # Appearances Chart
        st.write("### Appearances Over Time")
        top_domains_appearances = df_appearances.loc[top_domains.index]
        fig2 = go.Figure()
        for domain in top_domains_appearances.index:
            fig2.add_trace(go.Scatter(
                x=top_domains_appearances.columns,
                y=top_domains_appearances.loc[domain],
                mode="markers+lines",
                name=domain
            ))
        fig2.update_layout(
            title=f"Domain Appearances Over Time for Campaign '{selected_campaign}'",
            xaxis_title="Date",
            yaxis_title="Number of Appearances",
            updatemenus=[{"buttons": [{"args": [{"visible": True}], "label": "Show All", "method": "update"},
                                     {"args": [{"visible": "legendonly"}], "label": "Hide All", "method": "update"}],
                          "direction": "right", "showactive": True, "x": 1, "xanchor": "right", "y": 1.15, "yanchor": "top"}]
        )
        st.plotly_chart(fig2)
        st.write("### Additional Metrics Over Time")
        st.dataframe(df_metrics.style.format("{:.2f}"))
    else:
        st.write(f"No historical data available for the selected date range and campaign '{selected_campaign}'.")

elif page == "Campaign Management":
    st.header("Campaign Management")

    # Create New Campaign
    st.subheader("Create a New Campaign")
    campaign_id = st.text_input("Campaign ID (unique identifier)")
    campaign_name = st.text_input("Campaign Name")
    csv_file = st.file_uploader("Upload CSV File (job_title,location)", type=["csv"])

    if st.button("Create/Update Campaign"):
        if campaign_id and campaign_name and csv_file:
            if create_or_update_campaign(campaign_id, campaign_name, csv_file):
                st.success(f"Campaign '{campaign_id}' created/updated successfully!")
            else:
                st.error("Failed to create/update campaign. Please check the inputs.")
        else:
            st.error("Please fill in all fields and upload a CSV file.")

    # List Existing Campaigns
    st.subheader("Existing Campaigns")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT campaign_id, name, created_at FROM campaigns ORDER BY created_at DESC")
    campaigns = cursor.fetchall()
    cursor.close()
    conn.close()

    if campaigns:
        st.write("### Campaign List")
        for campaign in campaigns:
            st.write(f"- **Campaign ID:** {campaign[0]}, **Name:** {campaign[1]}, **Created At:** {campaign[2]}")
    else:
        st.write("No campaigns created yet.")

# GitHub workflow automation
if len(sys.argv) > 1 and sys.argv[1] == "github":
    print("🚀 Running automated fetch & store process (GitHub workflow) for default campaign")
    sov_data, appearances, avg_v_rank, avg_h_rank = compute_sov("default")
    save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, "default")
    print("✅ Data stored successfully for default campaign!")