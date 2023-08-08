from flask import Flask, render_template, request, url_for, flash, redirect
import os
import threading
from recognition import run
import concurrent.futures
import flask
from flask import jsonify
import pandas as pd
import json
import sqlite3
from Main import main
import shutil
import zipfile
import csv
import sys
from io import StringIO



thread = None
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SECRET_KEY'] = 'your secret key'

ALLOWED_EXTENSIONS = {'rar', 'zip'}

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('home'))

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('home'))

        if file and allowed_file(file.filename):
            current_dir = os.path.dirname(__file__)
            save_path = os.path.join(current_dir, 'static', 'resource', 'images.zip')
            file.save(save_path)
            flash('File uploaded successfully', 'success')
            return redirect(url_for('select'))

        flash('Invalid file format', 'error')

    return render_template("home.html")

@app.route("/select", methods=['GET', 'POST'])
def select():
    return render_template("select.html")

def process_job(job_type):
    if(job_type == 'test'):
        zip_path = './static/resource/images.zip' 
        target_folder = './static/resource/images'
        unzip_file(zip_path, target_folder)
        main()

def capture_output(thread):
    output_buffer = StringIO()
    sys.stdout = output_buffer

    # Wait for the thread to finish
    thread.result()

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Get captured output
    captured_output = output_buffer.getvalue()

# Global variable to store the captured output
global_captured_output = ""

def capture_output(thread):
    global global_captured_output
    
    output_buffer = StringIO()
    sys.stdout = output_buffer

    # Wait for the thread to finish
    thread.result()

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Get captured output
    global_captured_output = output_buffer.getvalue()

# Modify the `send_output_to_frontend` function to return the global captured output
def send_output_to_frontend():
    global global_captured_output
    return global_captured_output

@app.route("/check_thread_completion")
def check_thread_completion():
    global thread  
    if thread and thread.is_alive():
        completion_status = False
    else:
        completion_status = True

    response = jsonify({"completed": completion_status})
    return response

@app.route("/job_in_progress", methods=["GET", "POST"])
def job_in_progress():   
    return render_template("job_in_progress.html")

@app.route("/create_job", methods=["GET", "POST"])
def create_job():
    job_type = request.form['type']
    print(job_type)
    global thread 
    thread = threading.Thread(target=process_job, args=(job_type, ))
    thread.start()
        
    return redirect(url_for('job_in_progress'))


@app.route("/download_results")
def download_results():
    result_file_path = "predicted_labels.csv"

    response = flask.send_file(result_file_path, as_attachment=True)

    return response

def get_db_connection(): #Cette fonction get_db_connection() ouvre une connexion au fichier de base de données database.db pour executer du sql
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn 

@app.route("/label_display")
def label_display():
    df = pd.read_csv('predicted_labels.csv')

    image_paths = df['Image Name'].tolist()
    labels = df['Predicted Label'].tolist()
    MajorityVoting = df['MajorityVoting'].tolist()
    for i in range(len(image_paths)):
        image_paths[i] = 'static/resource/images/' + image_paths[i] + '.jpg'
    print(image_paths)
    conn = get_db_connection()
    for image, label, vote in zip(image_paths, labels, MajorityVoting):
        conn.execute('INSERT INTO Results (image, label, MajorityVoting) VALUES (?, ?, ?)',
                         (image,label,vote))
        conn.commit()
        print("The results have been stored")
    conn.close()
    return render_template("label_display.html", image_paths=image_paths, labels=labels, votes = MajorityVoting)

@app.route("/submit_modification", methods=["POST"])
def submit_modification():
    data = request.json
    print(data)

    # Store the modification data in the database
    try:
        # Establish a connection to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        for entry in data:
            modified_label = entry["modifiedLabel"]
            feedback = entry["feedback"]
            image_path = entry["imagePath"]

            # Get the result_id from the Results table based on the image_path
            sqlite_select_query = "SELECT result_id FROM Results WHERE image=?"
            cursor.execute(sqlite_select_query, (image_path,))
            record = cursor.fetchone()

            # Check if the record exists
            if record is not None:
                result_id = record[0]
                
                # Insert the feedback into the Feedback table
                conn.execute('INSERT INTO Feedback (label, feedback, result_id) VALUES (?, ?, ?)',
                             (modified_label, feedback, result_id))
                conn.commit()

        conn.close()
        return jsonify({"success": True})
    except sqlite3.Error as e:
        print("An error occurred:", e)
        # Rollback the transaction in case of an error
        conn.rollback()
        return jsonify({"success": False})


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def unzip_file(zip_path, target_folder):

    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        file_list = zip_ref.namelist()
        file_list = [f for f in file_list if not f.startswith('__MACOSX/')]


        for file in file_list:
            zip_ref.extract(file, target_folder)

    file_list.pop(0)

    for file in file_list:
        source_path = os.path.join(target_folder, file)
        destination_path = os.path.join(target_folder, os.path.basename(file))
        shutil.move(source_path, destination_path)

    print('unzip finished')

if __name__ == "__main__":
    app.run(debug=True)
