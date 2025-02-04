from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
import yfinance as yf
import plotly.express as px
import threading
import json
import logging

from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import NumPyMinimumEigensolver, SamplingVQE, QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Configuration
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # For development only
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Shared state
received_tickers = ["MSFT", "AAPL", "GOOG", "META", "IBM", "GS", "TSLA", "V", "UNH"]
tickers_lock = threading.Lock()

def download_stock_data():
    with tickers_lock:
        tickers = received_tickers.copy()
    
    try:
        data = yf.download(tickers, start="2021-01-01", end="2021-12-31")
        return data.xs('Close', level=0, axis=1), tickers
    except Exception as e:
        logging.error(f"Download error: {str(e)}")
        raise

def create_quadratic_program(mu, sigma, tickers):
    risk_factor = 0.5
    num_assets = len(tickers)
    budget = num_assets // 2

    qp = QuadraticProgram()
    for ticker in tickers:
        qp.binary_var(name=ticker)

    linear_objective = {ticker: -mu[i] for i, ticker in enumerate(tickers)}
    quadratic_objective = {
        (tickers[i], tickers[j]): risk_factor * sigma[i, j] 
        for i in range(num_assets) for j in range(num_assets)
    }
    
    qp.minimize(linear=linear_objective, quadratic=quadratic_objective)
    qp.linear_constraint(linear={ticker: 1 for ticker in tickers}, sense='==', rhs=budget)
    return qp

def run_optimization():
    try:
        close_prices, tickers = download_stock_data()
        log_returns = np.log(close_prices / close_prices.shift(1))
        mu = log_returns.mean().values
        sigma = log_returns.cov().values
        
        qp = create_quadratic_program(mu, sigma, tickers)
        
        results = []
        
        # Classical
        classical_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
        results.append({
            'method': 'Classical',
            'assets': [tickers[i] for i, x in enumerate(classical_result.x) if x > 0.5],
            'value': float(classical_result.fval)
        })
        
        # SamplingVQE
        ry_ansatz = TwoLocal(len(tickers), "ry", "cz", reps=3, entanglement="full")
        svqe_result = MinimumEigenOptimizer(SamplingVQE(Sampler(), ry_ansatz, COBYLA(maxiter=500))).solve(qp)
        results.append({
            'method': 'SamplingVQE',
            'assets': [tickers[i] for i, x in enumerate(svqe_result.x) if x > 0.5],
            'value': float(svqe_result.fval)
        })
        
        # QAOA
        qaoa_result = MinimumEigenOptimizer(QAOA(Sampler(), COBYLA(maxiter=250), reps=3)).solve(qp)
        results.append({
            'method': 'QAOA',
            'assets': [tickers[i] for i, x in enumerate(qaoa_result.x) if x > 0.5],
            'value': float(qaoa_result.fval)
        })
        
        # Plot data
        df = close_prices.reset_index().melt(id_vars='Date', var_name='Stock', value_name='Price')
        fig = px.line(df, x='Date', y='Price', color='Stock', template='plotly_white')
        fig.update_layout(hovermode='x unified')
        
        return {
            'results': results,
            'plot_html': fig.to_html(full_html=False, include_plotlyjs=False),
            'tickers': tickers
        }
    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}")
        return {'error': str(e)}

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to quantum backend'})

@socketio.on('update_tickers')
def handle_ticker_update(data):
    try:
        tickers = json.loads(data).get('tickers', [])
        with tickers_lock:
            global received_tickers
            received_tickers = tickers if tickers else received_tickers
        
        result = run_optimization()
        if 'error' in result:
            emit('error', {'message': result['error']})
        else:
            emit('update', {
                'results': result['results'],
                'plot': result['plot_html'],
                'tickers': result['tickers']
            })
    except Exception as e:
        emit('error', {'message': f"Update failed: {str(e)}"})





@socketio.on('update_tickers_list')
def handle_update_tickers_list(json_data):
    """
    Listens for the 'update_tickers_list' event from a React client.
    Expects a JSON payload with a key 'tickers' containing a list of symbols.
    The function updates the global received_tickers and runs the optimization.
    """
    try:
        data = json.loads(json_data)
        selected_tickers = data.get('tickers', [])
        print("Received tickers from React device:", selected_tickers)
        
        # Update the global received_tickers with the received list.
        with tickers_lock:
            global received_tickers
            if selected_tickers:
                received_tickers = selected_tickers
        
        # Run the optimization with the updated tickers.
        result = run_optimization()
        if 'error' in result:
            emit('error', {'message': result['error']})
        else:
            # Emit the updated optimization result back to the client.
            emit("tickers_received", {
                "tickers": result['tickers'],
                "results": result['results'],
                "plot": result['plot_html']
            })
    except Exception as e:
        print("Error processing update_tickers_list event:", e)
        emit("error", {"message": str(e)})





@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True,allow_unsafe_werkzeug=True)