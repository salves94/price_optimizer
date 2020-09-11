from functions import *
from flask import Flask, render_template, g, request, session, redirect, url_for
from database import get_db
from werkzeug.security import generate_password_hash, check_password_hash
import os

import math 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from matplotlib import animation, rc

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()

def get_current_user():
    user_result = None
    if 'email' in session:
        email = session['email']

        db = get_db()
        user_cur = db.execute('select id, full_name, email, password, admin from users where email = ?', [email])
        user_result = user_cur.fetchone()

    return user_result

@app.route('/')
def index():
    user = get_current_user()
    if user:
        return render_template('base_dashboard.html', email=user['email'])

    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    user = get_current_user()

    if request.method == 'POST':

        db = get_db()
        email = request.form['email']

        user_cur = db.execute('select id, full_name, email, password from users where email = ?', [email])
        existing_user = user_cur.fetchone()

        if existing_user:
            return render_template('register.html', user=user, error='User already exists!')

        hashed_password = generate_password_hash(request.form['password'], method='sha256')
        db.execute('insert into users (full_name, email, password, admin) values (?, ?, ?, ?)', [request.form['full_name'], request.form['email'], hashed_password, '0'])
        db.commit()
        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    user = get_current_user()
    error_password = None
    error_email = None

    if request.method == 'POST':
        db = get_db()

        email = request.form['email']
        password = request.form['password']

        user_cur = db.execute('select id, full_name, email, password from users where email = ?', [email])
        user_result = user_cur.fetchone()

        if user_result:
            if check_password_hash(user_result['password'], password):
                session['email'] = user_result['email']
                return redirect(url_for('index'))
            else:
                error_password = 'The password is incorrect.'
        else:
            error_email = 'The email is incorrect'

    return render_template('login.html', user=user, error_email=error_email, error_password=error_password)

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/price-optimization')
def viewPriceOptimization():
    user = get_current_user()
    if user:
        return render_template('price-optimization/view.html', email=user['email'])

    return redirect(url_for('login'))

@app.route('/price-optimization/create', methods=['GET', 'POST'])
def createPriceOptimization():
    user = get_current_user()
    if user:
        if request.method == 'POST':
            product_name = request.form['product_name']
            product_code = request.form['product_code']
            time_steps = int(request.form['time_steps'])
            max_price = int(request.form['max_price'])
            price_step = int(request.form['price_step'])
            q_0 = int(request.form['intercept_demand'])
            k = int(request.form['slope_demand'])
            unit_cost = int(request.form['product_cost'])
            increase_coefficient = int(request.form['increase_coefficient'])
            decrease_coefficient = int(request.form['decrease_coefficient'])
            gamma = int(request.form['gamma'])
            target_update = int(request.form['target_update'])
            batch_size = int(request.form['batch_size'])
            learning_rate = float(request.form['learning_rate'])
            num_episodes = int(request.form['num_episodes'])

            format_currency_v = np.vectorize(formatCurrency)

            price_grid = environmentSimulator(max_price, price_step, q_0, k, unit_cost, increase_coefficient, decrease_coefficient)

            optimal_constant_price = optimalConstantPrice(time_steps, price_grid, unit_cost, q_0, k, increase_coefficient, decrease_coefficient)
            constant_price = formatCurrency(optimal_constant_price['price'])
            constant_profit = formatCurrency(optimal_constant_price['profit'])

            optimal_sequence_of_prices = optimalSequenceOfPrices(optimal_constant_price, time_steps, price_grid, unit_cost, q_0, k, increase_coefficient, decrease_coefficient)
            sequence_prices = format_currency_v(optimal_sequence_of_prices['prices'])
            sequence_profit = formatCurrency(optimal_sequence_of_prices['profit'])

            best_profit_results = format_currency_v(deepQN(price_grid, time_steps, device, q_0, k, increase_coefficient, decrease_coefficient, unit_cost, gamma, target_update, batch_size, learning_rate, num_episodes))

            return render_template('price-optimization/results.html', email=user['email'], env_simulation_plot='/static/images/plot.png', constant_price=constant_price, constant_profit=constant_profit, optimal_seq_price_plot='/static/images/plot2.png', sequence_prices=sequence_prices, sequence_profit=sequence_profit, price_schedules='/static/images/plot3.png', returns_variation='/static/images/plot4.png', best_profit_results=best_profit_results)

        return render_template('price-optimization/create.html', email=user['email'])

    return redirect(url_for('login'))

    
if __name__ == '__main__':
    app.run(debug=True)