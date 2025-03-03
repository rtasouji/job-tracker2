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

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create or update share_of_voice table (using campaign_name)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS share_of_voice (
            id SERIAL PRIMARY KEY,
            domain TEXT NOT NULL,
            sov FLOAT NOT NULL,
            appearances INT DEFAULT 0,
            avg_v_rank FLOAT DEFAULT 0,
            avg_h_rank FLOAT DEFAULT 0,
            date DATE NOT NULL,
            campaign_name TEXT NOT NULL
        );
    """)

    # Create campaigns table to store campaign details and keywords with campaign_name as primary key
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_name TEXT PRIMARY KEY,
            job_titles TEXT NOT NULL,  -- JSON or string of job titles
            locations TEXT NOT NULL,   -- JSON or string of locations
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Check if campaign_id exists and rename it to campaign_name if necessary
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'share_of_voice' AND column_name = 'campaign_id'
        );
    """)
    if cursor.fetchone()[0]:
        cursor.execute("ALTER TABLE share_of_voice RENAME COLUMN campaign_id TO campaign_name;")
        logger.info("Renamed campaign_id to campaign_name in share_of_voice table")
    else:
        logger.info("No campaign_id column found in share_of_voice table; skipping rename")

    conn.commit()
    cursor.close()
    conn.close()
initialize_database()

# Load jobs (keywords and locations) from the database
def load_jobs(campaign_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT job_titles, locations FROM campaigns WHERE campaign_name = %s", (campaign_name,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        st.error(f"⚠️ Campaign '{campaign_name}' not found!")
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
        st.error(f"⚠️ Mismatch between job titles and locations for campaign '{campaign_name}'!")
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
def compute_sov(campaign_name):
    logger.info(f"Starting compute_sov for campaign {campaign_name}")
    domain_sov = defaultdict(float)
    domain_appearances = defaultdict(int)
    domain_v_rank = defaultdict(list)
    domain_h_rank = defaultdict(list)

    jobs_data = load_jobs(campaign_name)
    logger.info(f"Loaded {len(jobs_data)} job queries from database for campaign {campaign_name}")
    total_sov = 0  

    for job_query in jobs_data:
        job_title = job_query["job_title"]
        location = job_query["location"]
        logger.info(f"Processing job query: {job_title} in {location} for campaign {campaign_name}")

        try:
            jobs = get_google_jobs_results(job_title, location)
            logger.info(f"Retrieved {len(jobs)} job results for query in campaign {campaign_name}")
            if not jobs:
                logger.warning(f"No jobs found for query: {job_title} in {location} for campaign {campaign_name}")
            
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
            logger.error(f"Error processing job query {job_title} for campaign {campaign_name}: {str(e)}")
            continue

    if total_sov > 0:
        logger.info(f"Computed SoV for {len(domain_sov)} domains with total SoV: {total_sov} for campaign {campaign_name}")
        domain_sov = {domain: round((sov / total_sov) * 100, 4) for domain, sov in domain_sov.items()}
    else:
        logger.warning(f"No SoV computed for campaign {campaign_name} due to zero total SoV")
    
    domain_avg_v_rank = {domain: round(sum(vr) / len(vr), 2) for domain, vr in domain_v_rank.items() if vr}
    domain_avg_h_rank = {domain: round(sum(hr) / len(hr), 2) for domain, hr in domain_h_rank.items() if hr}

    logger.info(f"Returning SoV data for {len(domain_sov)} domains for campaign {campaign_name}")
    return domain_sov, domain_appearances, domain_avg_v_rank, domain_avg_h_rank

# Extract Domain from URL
def extract_domain(url):
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
    return domain.lower().replace("www.", "")

# Save data to database with campaign_name
def save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, campaign_name):
    logger.info(f"Attempting to save data for campaign {campaign_name}")
    if not sov_data:
        logger.warning(f"No SoV data to save to database for campaign {campaign_name}. Check if job results were retrieved.")
        return
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        today = datetime.date.today()
        logger.info(f"Saving data for date: {today} for campaign {campaign_name}")

        for domain in sov_data:
            logger.info(f"Inserting data for domain: {domain}, SoV: {sov_data[domain]}, Appearances: {appearances[domain]}, "
                        f"Avg V Rank: {avg_v_rank[domain]}, Avg H Rank: {avg_h_rank[domain]} for campaign {campaign_name}")
            cursor.execute("""
                INSERT INTO share_of_voice (domain, sov, appearances, avg_v_rank, avg_h_rank, date, campaign_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (domain, round(sov_data[domain], 2), appearances[domain], 
                  avg_v_rank[domain], avg_h_rank[domain], today, campaign_name))
        conn.commit()
        logger.info(f"Database commit successful for {len(sov_data)} domains in campaign {campaign_name}")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database error for campaign {campaign_name}: {str(e)}")
        raise

# Retrieve historical data for a specific campaign
def get_historical_data(start_date, end_date, campaign_name):
    logger.info(f"Retrieving historical data for campaign {campaign_name} from {start_date} to {end_date}")
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
        WHERE date BETWEEN %s AND %s AND campaign_name = %s
    """
    logger.info(f"Executing query for campaign {campaign_name} with date range {start_date} to {end_date}")
    cursor.execute(query, (start_date, end_date, campaign_name))
    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} rows for campaign {campaign_name}")

    df = pd.DataFrame(rows, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank"])

    cursor.close()
    conn.close()

    if df.empty:
        logger.warning(f"No data available for campaign {campaign_name} in the selected date range.")
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

    logger.info(f"Returning data for {len(df_sov)} domains for campaign {campaign_name}")
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
        SELECT domain, date, sov, appearances, avg_v_rank, avg_h_rank, campaign_name
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

    # Store total data with a special campaign_name (e.g., 'Total')
    save_to_db(domain_sov, domain_appearances, total_avg_v_rank, total_avg_h_rank, "Total")

# Create or update a campaign (store keywords in database using campaign_name)
def create_or_update_campaign(campaign_name, job_titles, locations):
    if not campaign_name or not job_titles or not locations:
        st.error("⚠️ Please provide a campaign name and at least one job title and location!")
        return False

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Convert lists to JSON strings for storage
        job_titles_json = json.dumps(job_titles)
        locations_json = json.dumps(locations)
        cursor.execute("""
            INSERT INTO campaigns (campaign_name, job_titles, locations)
            VALUES (%s, %s, %s)
            ON CONFLICT (campaign_name) DO UPDATE 
            SET job_titles = EXCLUDED.job_titles, locations = EXCLUDED.locations
        """, (campaign_name, job_titles_json, locations_json))
        conn.commit()
        logger.info(f"Campaign '{campaign_name}' created/updated successfully with {len(job_titles)} job titles and locations")
    except Exception as e:
        logger.error(f"Database error for campaign {campaign_name}: {str(e)}")
        conn.rollback()
        st.error(f"Database error: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()
    return True

# Delete a campaign and optionally its associated data
def delete_campaign(campaign_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Delete associated data from share_of_voice (optional, can be skipped for historical tracking)
        cursor.execute("DELETE FROM share_of_voice WHERE campaign_name = %s", (campaign_name,))
        logger.info(f"Deleted {cursor.rowcount} records from share_of_voice for campaign {campaign_name}")

        # Delete the campaign from campaigns table
        cursor.execute("DELETE FROM campaigns WHERE campaign_name = %s", (campaign_name,))
        if cursor.rowcount > 0:
            conn.commit()
            logger.info(f"Campaign '{campaign_name}' deleted successfully")
            st.success(f"Campaign '{campaign_name}' deleted successfully!")
        else:
            logger.warning(f"No campaign found with name '{campaign_name}'")
            st.warning(f"No campaign found with name '{campaign_name}'")
    except Exception as e:
        logger.error(f"Error deleting campaign {campaign_name}: {str(e)}")
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
    cursor.execute("SELECT campaign_name FROM campaigns")
    campaign_names = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    # Create a list of campaign names, including "Total" for all campaigns
    campaign_name_options = ["Total"]  # Total campaign label
    for name in campaign_names:
        campaign_name_options.append(name)

    # Dropdown with campaign names
    selected_campaign_name = st.sidebar.selectbox("Select Campaign", campaign_name_options, index=0)

    # Fetch & Store Data for the selected campaign or compute total
    if st.button("Fetch & Store Data"):
        if selected_campaign_name == "Total":
            compute_and_store_total_data()
            st.success("Total data across all campaigns stored successfully!")
        else:
            sov_data, appearances, avg_v_rank, avg_h_rank = compute_sov(selected_campaign_name)
            save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, selected_campaign_name)
            st.success(f"Data stored successfully for campaign '{selected_campaign_name}'!")

    # Show Historical Trends for the selected campaign or total
    st.write("### Visibility Over Time")
    if selected_campaign_name == "Total":
        df_sov, df_metrics, df_appearances = get_total_historical_data(start_date, end_date)
    else:
        df_sov, df_metrics, df_appearances = get_historical_data(start_date, end_date, selected_campaign_name)

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
            title=f"Domains Visibility Over Time for {selected_campaign_name}",
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
            title=f"Domain Appearances Over Time for {selected_campaign_name}",
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
        st.write(f"No historical data available for the selected date range and {selected_campaign_name}")

elif page == "Campaign Management":
    st.header("Campaign Management")

    # Create New Campaign
    st.subheader("Create a New Campaign")
    campaign_name = st.text_input("Campaign Name (unique identifier)")
    
    # Input for job titles and locations
    st.write("Add Job Titles and Locations:")
    job_titles = st.text_area("Job Titles (one per line)", height=100)
    locations = st.text_area("Locations (one per line, matching job titles)", height=100)

    if st.button("Create/Update Campaign"):
        if campaign_name:
            # Split input into lists, removing empty lines
            job_titles_list = [title.strip() for title in job_titles.split('\n') if title.strip()]
            locations_list = [loc.strip() for loc in locations.split('\n') if loc.strip()]
            
            if not job_titles_list or not locations_list:
                st.error("⚠️ Please provide at least one job title and location!")
            elif len(job_titles_list) != len(locations_list):
                st.error("⚠️ The number of job titles must match the number of locations!")
            else:
                if create_or_update_campaign(campaign_name, job_titles_list, locations_list):
                    st.success(f"Campaign '{campaign_name}' created/updated successfully!")
                else:
                    st.error("Failed to create/update campaign. Please check the inputs.")
        else:
            st.error("Please provide a campaign name and job titles and locations!")

    # Delete Campaign
    st.subheader("Delete a Campaign")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT campaign_name FROM campaigns")
    campaign_names = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    if campaign_names:
        st.write("### Select Campaign to Delete")
        selected_campaign_name = st.selectbox("Choose a campaign to delete", [""] + campaign_names)
        
        if selected_campaign_name:
            if st.button(f"Delete {selected_campaign_name}"):
                delete_campaign(selected_campaign_name)
                st.experimental_rerun()  # Refresh the page to reflect the deletion
    else:
        st.write("No campaigns available to delete.")

    # List Existing Campaigns
    st.subheader("Existing Campaigns")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT campaign_name, created_at FROM campaigns ORDER BY created_at DESC")
    campaigns = cursor.fetchall()
    cursor.close()
    conn.close()

    if campaigns:
        st.write("### Campaign List")
        for campaign_name, created_at in campaigns:
            st.write(f"- **Campaign Name:** {campaign_name}, **Created At:** {created_at}")
    else:
        st.write("No campaigns created yet.")

# GitHub workflow automation (updated to use campaign names)
if len(sys.argv) > 1 and sys.argv[1] == "github":
    print("🚀 Running automated fetch & store process (GitHub workflow) for default and total campaigns")
    # Process default campaign (using 'Default' as the name)
    sov_data, appearances, avg_v_rank, avg_h_rank = compute_sov("Default")
    save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, "Default")
    print("✅ Data stored successfully for default campaign!")
    
    # Process total campaign
    compute_and_store_total_data()
    print("✅ Total data across all campaigns stored successfully!")
