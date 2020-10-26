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
import sqlite3
from flask import jsonify
from flask_babel import Babel, format_date, gettext

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
babel = Babel(app)
ROOT = 'static/images/'

@babel.localeselector
def get_locale():
    language = 'en'
    if 'email' in session:
        email = session['email']

        try:
            sqliteConnection = get_db()
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")

            sql_select_query = """select language from users where email = ?"""
            cursor.execute(sql_select_query, (email,))
            result = cursor.fetchone()
            language = result['language']
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)

    return language

@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()

def get_current_user():
    user_result = None
    if 'email' in session:
        email = session['email']

        try:
            sqliteConnection = get_db()
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")

            sql_select_query = """select * from users where email = ?"""
            cursor.execute(sql_select_query, (email,))
            user_result = cursor.fetchone()
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)

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
        db.execute('insert into users (full_name, email, password, language) values (?, ?, ?, ?)', [request.form['full_name'], request.form['email'], hashed_password, request.form['language']])
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

        user_cur = db.execute('select * from users where email = ?', [email])
        user_result = user_cur.fetchone()

        if user_result:
            if check_password_hash(user_result['password'], password):
                session['email'] = user_result['email']
                return redirect(url_for('index'))
            else:
                error = gettext('Email or password is incorrect.')
        else:
            error = gettext('Email or password is incorrect.')

    return render_template('login.html', user=user, error=error)

@app.route('/language/<language>')
def setLanguage(language):

    email = session['email']
    try:
        sqliteConnection = get_db()
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_update_query = """Update users set language = ? where email = ?"""
        columnValues = (language, email)
        cursor.execute(sqlite_update_query, columnValues)
        sqliteConnection.commit()
        print("Total", cursor.rowcount, "Records updated successfully")
        sqliteConnection.commit()
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to update multiple records of sqlite table", error)

    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/price-optimization/create', methods=['GET', 'POST'])
def createPriceOptimization():
    user = get_current_user()
    if user:
        if request.method == 'POST':
            product_id = request.form['product_id']
            time_steps = int(request.form['time_steps'])
            max_price = float(request.form['max_price'])
            price_step = float(request.form['price_step'])
            q_0 = float(request.form['intercept_demand'])
            k = float(request.form['slope_demand'])
            unit_cost = float(request.form['product_cost'])
            increase_coefficient = float(request.form['increase_coefficient'])
            decrease_coefficient = float(request.form['decrease_coefficient'])
            gamma = float(request.form['gamma'])
            target_update = float(request.form['target_update'])
            batch_size = int(request.form['batch_size'])
            learning_rate = float(request.form['learning_rate'])
            num_episodes = int(request.form['num_episodes'])

            try:
                sqliteConnection = get_db()
                cursor = sqliteConnection.cursor()
                print("Connected to SQLite")

                sql_select_query = "insert into price_optimization_inputs (product_id, time_steps, max_price, price_step, q_0, k, unit_cost, increase_coefficient, decrease_coefficient, gamma, target_update, batch_size, learning_rate, num_episodes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                cursor.execute(sql_select_query, (product_id, time_steps, max_price, price_step, q_0, k, unit_cost, increase_coefficient, decrease_coefficient, gamma, target_update, batch_size, learning_rate, num_episodes))
                sqliteConnection.commit()
                price_optimization_inputs_id = cursor.lastrowid

            except sqlite3.Error as error:
                sqliteConnection.rollback()
                print(error)

            env_simulation_src = ROOT + str(price_optimization_inputs_id) + '-environment-simulation.png'
            optimal_seq_price_src = ROOT + str(price_optimization_inputs_id) + '-price-sequence.png'
            returns_variation_src = ROOT + str(price_optimization_inputs_id) + '-returns-variation.png'
            price_schedules_src = ROOT + str(price_optimization_inputs_id) + '-price-schedules.png'
            td_errors_src = ROOT + str(price_optimization_inputs_id) + '-td-errors.png'
            correlation_src = ROOT + str(price_optimization_inputs_id) + '-correlation.png'

            format_currency_v = np.vectorize(formatCurrency)

            price_grid = environmentSimulator(max_price, price_step, q_0, k, unit_cost, increase_coefficient, decrease_coefficient, env_simulation_src)


            optimal_constant_price = optimalConstantPrice(time_steps, price_grid, unit_cost, q_0, k, increase_coefficient, decrease_coefficient)
            constant_price = formatCurrency(optimal_constant_price['price'])
            constant_profit = formatCurrency(optimal_constant_price['profit'])


            optimal_sequence_of_prices = optimalSequenceOfPrices(optimal_constant_price, time_steps, price_grid, unit_cost, q_0, k, increase_coefficient, decrease_coefficient, optimal_seq_price_src)
            sequence_prices = format_currency_v(optimal_sequence_of_prices['prices'])
            sequence_profit = formatCurrency(optimal_sequence_of_prices['profit'])


            profit_results = deepQN(price_grid, time_steps, device, q_0, k, increase_coefficient, decrease_coefficient, unit_cost, gamma, target_update, batch_size, learning_rate, num_episodes, returns_variation_src, price_schedules_src)
            best_profit_results = format_currency_v(profit_results['profit'])


            q_trace = TDError(gamma, device, profit_results['memory'], profit_results['policy_net'], profit_results['target_net'], td_errors_src)


            correlation(time_steps, gamma, profit_results['policy_net'], profit_results['policy'], price_grid, q_0, k, increase_coefficient, decrease_coefficient, unit_cost, correlation_src)

            try:
                sqliteConnection = get_db()
                cursor = sqliteConnection.cursor()
                print("Connected to SQLite")

                sql_select_query = "insert into price_optimization_results (price_optimization_inputs_id, constant_price, constant_profit, sequence_prices, sequence_profit, best_profit_results, env_simulation_src, optimal_seq_price_src, returns_variation_src, price_schedules_src, td_errors_src, correlation_src) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                cursor.execute(sql_select_query, (price_optimization_inputs_id, constant_price, constant_profit, str(sequence_prices), sequence_profit, str(best_profit_results), env_simulation_src, optimal_seq_price_src, returns_variation_src, price_schedules_src, td_errors_src, correlation_src))
                sqliteConnection.commit()
                price_optimization_results_id = cursor.lastrowid

            except sqlite3.Error as error:
                sqliteConnection.rollback()
                print(error)

            return redirect(url_for('viewPriceOptimizationId', price_optimization_results_id=price_optimization_results_id))

        try:
            sqliteConnection = get_db()
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")

            sql_select_query = """select * from products"""
            cursor.execute(sql_select_query)
            products = cursor.fetchall()
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)

        return render_template('price-optimization/create.html', email=user['email'], products=products)

    return redirect(url_for('login'))

@app.route('/price-optimization/<price_optimization_results_id>')
def viewPriceOptimizationId(price_optimization_results_id):
    user = get_current_user()
    if user:
        try:
            sqliteConnection = get_db()
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")

            sql_select_query = "select * from price_optimization_results where id = ?"
            cursor.execute(sql_select_query, (price_optimization_results_id,))
            price_optimization_results = cursor.fetchone()

            sql_select_query = "select * from price_optimization_inputs where id = ?"
            cursor.execute(sql_select_query, (price_optimization_results['price_optimization_inputs_id'],))
            price_optimization_inputs = cursor.fetchone()

            cursor.close()

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)

        sequence_prices = price_optimization_results['sequence_prices'][1:-1]
        best_profit_results = price_optimization_results['best_profit_results'][1:-1]

        sequence_prices = sequence_prices.replace("'", '').split(' ')
        best_profit_results = best_profit_results.replace("'", '').split(' ')

        return render_template('price-optimization/results.html', email=user['email'], price_optimization_results=price_optimization_results, price_optimization_inputs=price_optimization_inputs, sequence_prices=sequence_prices, best_profit_results=best_profit_results)

    return redirect(url_for('login'))

@app.route('/price-optimization')
def viewPriceOptimization():
    user = get_current_user()
    if user:

        try:
            sqliteConnection = get_db()
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")

            sql_select_query = "select price_optimization_results.id as results_id, price_optimization_results.price_optimization_inputs_id as inputs_id, price_optimization_inputs.product_id, products.name, products.sku from price_optimization_results inner join price_optimization_inputs on price_optimization_results.price_optimization_inputs_id = price_optimization_inputs.id inner join products on price_optimization_inputs.product_id = products.id"
            cursor.execute(sql_select_query)
            price_optimization = cursor.fetchall()
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)

        return render_template('price-optimization/view.html', email=user['email'], price_optimization=price_optimization)

    return redirect(url_for('login'))

@app.route('/price-optimization/delete/<id>')
def deletePriceOptimization(id):
    user = get_current_user()
    if user:
        try:
            sqliteConnection = get_db()
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")

            sql_select_query = "select price_optimization_inputs_id, env_simulation_src, optimal_seq_price_src, returns_variation_src, price_schedules_src, td_errors_src, correlation_src from price_optimization_results where id = ?"
            cursor.execute(sql_select_query, (id,))
            price_optimization_results = cursor.fetchone()

            if os.path.exists(price_optimization_results['env_simulation_src']):
                os.remove(price_optimization_results['env_simulation_src'])
            else:
                print("The file does not exist")

            if os.path.exists(price_optimization_results['optimal_seq_price_src']):
                os.remove(price_optimization_results['optimal_seq_price_src'])
            else:
                print("The file does not exist")

            if os.path.exists(price_optimization_results['returns_variation_src']):
                os.remove(price_optimization_results['returns_variation_src'])
            else:
                print("The file does not exist")

            if os.path.exists(price_optimization_results['price_schedules_src']):
                os.remove(price_optimization_results['price_schedules_src'])
            else:
                print("The file does not exist")

            if os.path.exists(price_optimization_results['td_errors_src']):
                os.remove(price_optimization_results['td_errors_src'])
            else:
                print("The file does not exist")

            if os.path.exists(price_optimization_results['correlation_src']):
                os.remove(price_optimization_results['correlation_src'])
            else:
                print("The file does not exist")

            sql_select_query = "PRAGMA foreign_keys = ON;"
            cursor.execute(sql_select_query)

            sql_delete_query = "DELETE from price_optimization_inputs where id = ?"
            cursor.execute(sql_delete_query, [price_optimization_results['price_optimization_inputs_id']])
            sqliteConnection.commit()
            print("Record deleted successfully ")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)

        return redirect(url_for('viewPriceOptimization'))

    return redirect(url_for('login'))

@app.route('/products')
def viewProducts():
    user = get_current_user()
    products = ''
    if user:

        try:
            sqliteConnection = get_db()
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")

            sql_select_query = "select * from products"
            cursor.execute(sql_select_query)
            products = cursor.fetchall()
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)

        return render_template('products/view.html', email=user['email'], products=products)

    return redirect(url_for('login'))

@app.route('/products/create', methods=['GET', 'POST'])
def createProduct():
    user = get_current_user()
    if user:
        if request.method == 'POST':
            name = request.form['name']
            sku = request.form['sku']

            db = get_db()

            product = db.execute('select name, sku from products where sku = ?', [sku])
            existing_product = product.fetchone()

            if existing_product:
                return render_template('products/create.html', product=product, error='Product already exists!')

            db.execute('insert into products (name, sku) values (?, ?)', [name, sku])
            db.commit()

            return redirect(url_for('viewProducts'))

        return render_template('products/create.html', email=user['email'])

    return redirect(url_for('login'))

@app.route('/products/edit', methods=['POST'])
def editProduct():
    user = get_current_user()
    if user:
        id = request.form['id']
        if (request.form['action'] == 'delete'):
            try:
                sqliteConnection = get_db()
                cursor = sqliteConnection.cursor()
                print("Connected to SQLite")

                sql_select_query = "PRAGMA foreign_keys = ON;"
                cursor.execute(sql_select_query)

                sql_delete_query = "DELETE from products where id = ?"
                cursor.execute(sql_delete_query, [id])
                sqliteConnection.commit()
                print("Record deleted successfully ")
                cursor.close()

            except sqlite3.Error as error:
                print("Failed to delete record from sqlite table", error)

            finally:
                return jsonify(action='delete', id=id)

        if (request.form['action'] == 'edit'):
            name = request.form['name']
            sku = request.form['sku']

            try:
                sqliteConnection = get_db()
                cursor = sqliteConnection.cursor()
                print("Connected to SQLite")

                sqlite_update_query = """Update products set name = ?, sku = ? where id = ?"""
                columnValues = (name, sku, id)
                cursor.execute(sqlite_update_query, columnValues)
                sqliteConnection.commit()
                print("Total", cursor.rowcount, "Records updated successfully")
                sqliteConnection.commit()
                cursor.close()

            except sqlite3.Error as error:
                print("Failed to update multiple records of sqlite table", error)

            finally:
                return jsonify(action='edit', id=id)

    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
