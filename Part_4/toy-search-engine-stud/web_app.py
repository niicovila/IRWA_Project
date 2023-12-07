from datetime import datetime
from flask import Flask, request, render_template, session, redirect, url_for, jsonify, abort, url_for, redirect
import httpagentparser
from myapp.core.utils import get_tweet_info  # for getting the user agent as json
from myapp.search.search_engine import SearchEngine
import os
from myapp.search.load_corpus import load_corpus
from json import JSONEncoder
import uuid
from myapp.data_collection.data_models import Session_Data, Click_Data, Request_Data, format_user_agent
from myapp.data_collection.data_storage import DataStorage
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta
from collections import Counter
import numpy as np

app = Flask(__name__)

# Initialize data storage
storage = DataStorage()

app.secret_key = 'your_secret_key'
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'

# Load corpus and instantiate the search engine
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
json_file_path = path + "/Rus_Ukr_war_data.json"
df_file_path = path + "/embeddings_df.csv"

tweets, combined_df = load_corpus(json_file_path, df_file_path)
search_engine = SearchEngine()

@app.route('/')
def index():
    user_ip = request.remote_addr
    current_time = datetime.now()

    # Check if this IP already has an ongoing session
    existing_session = next((s for s in storage.sessions.values() if s.user_ip == user_ip and not s.end_time), None)

    if not existing_session:
        # Create a new session
        session_id = str(uuid.uuid4())
        user_agent = httpagentparser.detect(request.headers.get('User-Agent'))
        formatted_user_agent = format_user_agent(user_agent)
        new_session = Session_Data(session_id, user_ip, current_time, user_agent=formatted_user_agent)
        storage.add_session(new_session)
        session['session_id'] = session_id
    else:
        # Use the existing session ID
        session_id = existing_session.session_id
        session['session_id'] = session_id

    return render_template('index.html', page_title="Welcome")

@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    session['last_search_query'] = search_query
    return redirect(url_for('search_results', query=search_query, page=1))

@app.route('/search_results/<query>/<int:page>')
def search_results(query, page):
    search_query = session.get('last_search_query', query)
    request_id = str(uuid.uuid4())
    session_id = session.get('session_id', None)
    storage.add_request(Request_Data(request_id, session_id, search_query, datetime.now()))

    results = search_engine.search(search_query, request_id, combined_df, tweets)
    per_page = 10
    paginated_results = results[(page-1)*per_page : page*per_page]

    found_count = len(results)
    pages = -(-found_count // per_page)  # Ceiling division to calculate total pages

    return render_template('results.html', results_list=paginated_results, page_title="Results", found_counter=found_count, pages=pages, current_page=page)

@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    # Fetching session and request data from storage
    sessions = list(storage.sessions.values())  
    requests = list(storage.requests.values())  
    clicks = list(storage.clicks.values())

    sessions_plot = plot_sessions_per_time(sessions)
    requests_plot = plot_requests_per_time(requests)
    clicks_plot = plot_clicks_per_time(clicks)

    # Example data for charts (you should replace these with your actual data)
    session_times = [s.start_time for s in sessions]
    request_counts = [len(requests), len(clicks)]  # Example data

    # Create charts
    create_chart(session_times, "Session Times", "Sessions", "Time", "static/session_times.png")
    create_chart(request_counts, "Requests and Clicks", "Type", "Count", "static/requests_clicks.png")

    # Pass the URLs of the saved images to the template
    session_chart_url = url_for('static', filename='session_times.png')
    requests_clicks_chart_url = url_for('static', filename='requests_clicks.png')
   
    # Example: Clicks per document
    clicks_per_document = Counter([click.document_id for click in clicks])
    create_bar_chart(clicks_per_document, "Clicks per Document", "Document ID", "Clicks", "static/clicks_per_document.png")

    # Example: Requests over time
    requests_over_time = [r.timestamp for r in requests]
    create_line_chart(requests_over_time, "Requests Over Time", "Time", "Number of Requests", "static/requests_over_time.png")

    # URLs for new charts
    clicks_per_document_chart_url = url_for('static', filename='clicks_per_document.png')
    requests_over_time_chart_url = url_for('static', filename='requests_over_time.png')

    # You can also add any additional processing of the data here if needed
    return render_template('analytics.html',
                           sessions=sessions,
                           requests=requests,
                           clicks=clicks,
                           session_chart_url=session_chart_url,
                           requests_clicks_chart_url=requests_clicks_chart_url,
                           clicks_per_document_chart_url=clicks_per_document_chart_url,
                           requests_over_time_chart_url=requests_over_time_chart_url, sessions_plot=sessions_plot, requests_plot=requests_plot, clicks_plot=clicks_plot)

@app.route('/tweet/<tweet_id>')
def tweet_detail(tweet_id):
    # Fetch the specific tweet details
    tweet = next((t for t in tweets if str(t['id']) == tweet_id), None)
    tweet_info= get_tweet_info(tweet)
    if tweet is None:
        abort(404)  # If the tweet is not found, return a 404 error

    return render_template('tweet_detail.html', tweet=tweet_info)

@app.route('/track_click/<tweet_id>')
def track_click(tweet_id):
    query = request.args.get('query')
    rank = request.args.get('rank')
    current_time = datetime.now()
    
    # Record the click data
    click_id = str(uuid.uuid4())
    session_id = session.get('session_id', None)
    storage.add_click(Click_Data(click_id, session_id, tweet_id, current_time, query=query, rank=rank))

    # Redirect to the tweet detail page
    return redirect(url_for('tweet_detail', tweet_id=tweet_id))

@app.route('/update_plots', methods=['POST'])
def update_plots():
    time_unit = request.json['time_unit']
    sessions = list(storage.sessions.values())
    requests = list(storage.requests.values())
    clicks = list(storage.clicks.values())

    # Generate the sessions plot based on the selected time range
    sessions_plot = plot_sessions_per_time(sessions, time_unit=time_unit)
    requests_plot = plot_requests_per_time(requests, time_unit=time_unit)
    clicks_plot = plot_clicks_per_time(clicks, time_unit=time_unit)

    return jsonify({'sessions_plot': sessions_plot, 'requests_plot': requests_plot, 'clicks_plot': clicks_plot})


def plot_sessions_per_time(sessions, time_unit='hour'):
    if not sessions:
        return None

    end_time = max(session.end_time or datetime.now() for session in sessions)

    if time_unit == 'hour':
        start_time = end_time - timedelta(minutes=60)
        time_range = [start_time + timedelta(minutes=i) for i in range(61)] # 60 minutes in minutes
    elif time_unit == 'day':
        start_time = end_time - timedelta(hours=24)
        time_range = [start_time + timedelta(minutes=i) for i in range(1441)]  # 24 hours in minutes
    elif time_unit == 'week':
        start_time = end_time - timedelta(days=7)
        time_range = [start_time + timedelta(hours=i) for i in range(169)]  # 7 days in hours
    else:
        raise ValueError(f"Invalid time unit: {time_unit}")

    session_counts = [sum(1 for s in sessions if s.start_time <= time < (s.end_time or datetime.now())) for time in time_range]

    # Check if there is an active session at the end time
    active_session = next((s for s in sessions if s.start_time <= end_time and (s.end_time is None or s.end_time > end_time)), None)
    if active_session and session_counts[-1] == 0:
        print(active_session)
        session_counts[-1] = 1

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(time_range, session_counts, marker='o')
    plt.title(f'Sessions in Last {time_unit.capitalize()}')
    if time_unit == 'hour' or time_unit == 'day':
        plt.xlabel('Hour')
    elif time_unit == 'week':
        plt.xlabel('Day')
    else:
        raise ValueError(f"Invalid time unit: {time_unit}")
    plt.ylabel('Number of Sessions')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert the plot to a base64-encoded image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return plot_image

def plot_requests_per_time(requests, time_unit='hour'):
    if not requests:
        return None

    end_time = max(request.timestamp for request in requests)

    if time_unit == 'hour':
        start_time = end_time - timedelta(minutes=60)
        time_range = [start_time + timedelta(minutes=i) for i in range(61)]  # 60 minutes in minutes
    elif time_unit == 'day':
        start_time = end_time - timedelta(hours=24)
        time_range = [start_time + timedelta(minutes=i) for i in range(1441)]  # 24 hours in minutes
    elif time_unit == 'week':
        start_time = end_time - timedelta(days=7)
        time_range = [start_time + timedelta(hours=i) for i in range(169)]  # 7 days in hours
    else:
        raise ValueError(f"Invalid time unit: {time_unit}")

    request_counts = [sum(1 for r in requests if start_time <= r.timestamp < start_time + timedelta(minutes=1)) for start_time in time_range]

    active_events = [r.timestamp for r in requests if start_time <= r.timestamp <= start_time + timedelta(minutes=1)]
    if not active_events and request_counts[-1] == 0:
        request_counts[-1] = 0  # Set to 0 if there are no requests in the last minute and the last value is 0
    elif active_events:
        request_counts[-1] = len(active_events)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(time_range, request_counts, marker='o')
    plt.title(f'Requests Per {time_unit.capitalize()}')
    plt.xlabel(f'{time_unit.capitalize()}')
    plt.ylabel('Number of Requests')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert the plot to a base64-encoded image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return plot_image

def plot_clicks_per_time(clicks, time_unit='hour'):
    if not clicks:
        return None

    end_time = max(click.timestamp for click in clicks)

    if time_unit == 'hour':
        start_time = end_time - timedelta(minutes=60)
        time_range = [start_time + timedelta(minutes=i) for i in range(61)]  # 60 minutes in minutes
    elif time_unit == 'day':
        start_time = end_time - timedelta(hours=24)
        time_range = [start_time + timedelta(minutes=i) for i in range(1441)]  # 24 hours in minutes
    elif time_unit == 'week':
        start_time = end_time - timedelta(days=7)
        time_range = [start_time + timedelta(hours=i) for i in range(169)]  # 7 days in hours
    else:
        raise ValueError(f"Invalid time unit: {time_unit}")

    click_counts = [sum(1 for c in clicks if start_time <= c.timestamp < start_time + timedelta(minutes=1)) for start_time in time_range]

    active_events = [r.timestamp for r in clicks if start_time <= r.timestamp <= start_time + timedelta(minutes=1)]
    if not active_events and click_counts[-1] == 0:
        click_counts[-1] = 0  
    elif active_events:
        click_counts[-1] = len(active_events)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(time_range, click_counts, marker='o')
    plt.title(f'Clicks Per {time_unit.capitalize()}')
    plt.xlabel(f'{time_unit.capitalize()}')
    plt.ylabel('Number of Clicks')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert the plot to a base64-encoded image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return plot_image

def create_line_chart(data, title, x_label, y_label, filename):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.close()

def create_bar_chart(data, title, x_label, y_label, filename):
    plt.figure()
    positions = np.arange(len(data))
    plt.bar(positions, list(data.values()), align='center')
    plt.xticks(positions, list(data.keys()))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.close()

def create_chart(data, title, x_label, y_label, filename):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=True)
