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
            return render_template('register.html', user=user, error='Email already exists!')

        hashed_password = generate_password_hash(request.form['password'], method='sha256')
        db.execute('insert into users (full_name, email, password, admin) values (?, ?, ?, ?)', [request.form['full_name'], request.form['email'], hashed_password, '0'])
        db.commit()
        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    user = get_current_user()
    error = None

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
                error = 'Email or password is incorrect.'
        else:
            error = 'Email or password is incorrect.'

    return render_template('login.html', user=user, error=error)

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
            product_id = request.form['product_id']
            time_steps = int(request.form['time_steps'])
            max_price = int(request.form['max_price'])
            price_step = int(request.form['price_step'])
            q_0 = int(request.form['intercept_demand'])
            k = int(request.form['slope_demand'])
            unit_cost = int(request.form['product_cost'])
            increase_coefficient = int(request.form['increase_coefficient'])
            decrease_coefficient = int(request.form['decrease_coefficient'])
            gamma = float(request.form['gamma'])
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


            profit_results = deepQN(price_grid, time_steps, device, q_0, k, increase_coefficient, decrease_coefficient, unit_cost, gamma, target_update, batch_size, learning_rate, num_episodes)
            best_profit_results = format_currency_v(profit_results['profit'])


            q_trace = TDError(gamma, device, profit_results['memory'], profit_results['policy_net'], profit_results['target_net'])


            correlation(time_steps, gamma, profit_results['policy_net'], profit_results['policy'], price_grid, q_0, k, increase_coefficient, decrease_coefficient, unit_cost)

            return render_template('price-optimization/results.html', email=user['email'], env_simulation_plot='/static/images/plot.png', constant_price=constant_price, constant_profit=constant_profit, optimal_seq_price_plot='/static/images/plot2.png', sequence_prices=sequence_prices, sequence_profit=sequence_profit, price_schedules='/static/images/plot3.png', returns_variation='/static/images/plot4.png', best_profit_results=best_profit_results, q_trace=q_trace, td_errors='/static/images/asd.png', correlation='/static/images/plot6.png')

        return render_template('price-optimization/create.html', email=user['email'])

    return redirect(url_for('login'))

@app.route('/products')
def viewProducts():
    user = get_current_user()
    if user:
        db = get_db()
        products = db.execute('select id, name, sku from products')
        return render_template('products/view.html', email=user['email'], products=products)

    return redirect(url_for('login'))

@app.route('/products/create', methods=['GET', 'POST'])
def createProduct():
    user = get_current_user()
    if user:
        if request.method == 'POST':
            name = request.form['name']
            sku = request.form['sku']
            description = request.form['description']

            db = get_db()

            product = db.execute('select name, sku from products where sku = ?', [sku])
            existing_product = product.fetchone()

            if existing_product:
                return render_template('products/create.html', product=product, error='Product already exists!')

            db.execute('insert into products (name, sku, description) values (?, ?, ?)', [name, sku, description])
            db.commit()

            return redirect(url_for('viewProducts'))

        return render_template('products/create.html', email=user['email'])

    return redirect(url_for('login'))

@app.route('/products/edit', methods=['POST'])
def editProduct():
    user = get_current_user()
    if user:
        print(request.form['name'])
        """
        id = request.form['id']
        name = request.form['name']
        sku = request.form['sku'] 

        db = get_db()

        product = db.execute('select name, sku from products where sku = ?', [sku])
        existing_product = product.fetchone()

        if existing_product:
            return render_template('products/create.html', product=product, error='Product already exists!')

        db.execute('insert into products (name, sku, description) values (?, ?, ?)', [name, sku, description])
        db.commit()
"""
        return redirect(url_for('viewProducts'))
    return redirect(url_for('login'))
    
if __name__ == '__main__':
    app.run(debug=True)