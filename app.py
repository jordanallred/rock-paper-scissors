from flask import Flask, render_template, request, session, jsonify
import random
import pickle
import base64
import numpy as np
import os
import tempfile
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()
app.config['SESSION_PERMANENT'] = False

try:
    from flask_session import Session

    Session(app)
except ImportError:
    print("WARNING: Flask-Session not installed. Large sessions will fail.")
    print("Install with: pip install Flask-Session")

CHOICES = ['rock', 'paper', 'scissors']

BEATS = {
    'rock': 'scissors',
    'paper': 'rock',
    'scissors': 'paper'
}

COUNTER_TO = {
    'rock': 'paper',
    'paper': 'scissors',
    'scissors': 'rock'
}


class ImprovedMLPredictor:
    """A predictor that prioritizes learning from data with minimal human intervention."""

    def __init__(self, n_gram_range=(1, 7), confidence_threshold=0.55):
        """Initialize the predictor with configurable n-gram lengths.

        Args:
            n_gram_range (tuple): The minimum and maximum n-gram lengths to track (inclusive)
            confidence_threshold (float): Threshold to determine when to trust ML vs random
        """
        self.min_n, self.max_n = n_gram_range
        self.n_values = list(range(self.min_n, self.max_n + 1))
        self.confidence_threshold = confidence_threshold

        self.ngram_weights = {n: n ** 1.5 for n in self.n_values}

        self.history = []
        self.transitions = {n: {} for n in self.n_values}
        self.move_counts = {'rock': 0, 'paper': 0, 'scissors': 0}

        self.prediction_accuracy = {'correct': 0, 'total': 0}
        self.randomness_used = 0
        self.total_decisions = 0

    def update(self, move, result=None):
        """Update the model with a new move.

        Args:
            move (str): The user's move ('rock', 'paper', or 'scissors')
            result (str, optional): The result of the round (unused in pure ML approach)
        """

        self.history.append(move)
        self.move_counts[move] += 1

        for n in self.n_values:
            self._update_ngram(n, move)

    def _update_ngram(self, n, move):
        """Update a specific n-gram transition table.

        Args:
            n (int): The n-gram length
            move (str): The user's move
        """
        if len(self.history) <= n:
            return

        context = tuple(self.history[-(n + 1):-1]) if n > 0 else ()

        if context not in self.transitions[n]:
            self.transitions[n][context] = {'rock': 0, 'paper': 0, 'scissors': 0}

        self.transitions[n][context][move] += 1

    def _get_next_move_distribution(self, n, context):
        """Get the probability distribution for the next move given a context.

        Args:
            n (int): The n-gram length
            context (tuple): The context to check

        Returns:
            dict: Probability distribution for next move
            float: Total count of observations
        """
        if context not in self.transitions[n]:
            return None, 0

        counts = self.transitions[n][context]
        total = sum(counts.values())

        if total == 0:
            return None, 0

        return {move: count / total for move, count in counts.items()}, total

    def predict(self):
        """Predict the user's next move based purely on observed patterns.

        Returns:
            str: Predicted move ('rock', 'paper', or 'scissors')
            float: Confidence in the prediction (0-1)
            str: Strategy used for prediction
        """

        predictions = []

        for n in sorted(self.n_values, reverse=True):
            if len(self.history) < n:
                continue

            context = tuple(self.history[-n:]) if n > 0 else ()

            distribution, observations = self._get_next_move_distribution(n, context)

            if distribution:
                predicted_move = max(distribution, key=distribution.get)
                confidence = distribution[predicted_move]
                weight = self.ngram_weights.get(n, 1.0) * (min(observations, 10) / 10)

                predictions.append({
                    'move': predicted_move,
                    'confidence': confidence,
                    'weight': weight,
                    'strategy': f"{n}-gram pattern" if n > 0 else "Base frequency",
                    'observations': observations
                })

        if not predictions:
            return random.choice(CHOICES), 0.0, "Random (insufficient data)"

        best_prediction = max(predictions, key=lambda x: x['confidence'] * x['weight'])

        return best_prediction['move'], best_prediction['confidence'], best_prediction['strategy']

    def get_counter_move(self):
        """Choose a move that counters the predicted user move.

        Returns:
            str: AI's move ('rock', 'paper', or 'scissors')
            float: Confidence in the prediction (0-1)
            str: Strategy used
        """
        self.total_decisions += 1

        predicted_move, confidence, strategy = self.predict()

        if confidence < self.confidence_threshold:
            self.randomness_used += 1
            return random.choice(CHOICES), confidence, "Random (low confidence)"

        return COUNTER_TO[predicted_move], confidence, strategy

    def get_transition_heatmap(self, n=None):
        """Get transition data for visualization.

        Args:
            n (int, optional): Specific n-gram length to visualize. If None, uses middle value.

        Returns:
            dict: Transition counts for heatmap visualization
        """
        result = {}

        if n is None:
            n = self.n_values[len(self.n_values) // 2]

        if n not in self.transitions:
            return {}

        for context, next_moves in self.transitions[n].items():
            if len(context) == 0:
                context_str = "Overall"
            else:
                context_str = " → ".join(context)
            result[context_str] = next_moves

        return result

    def get_move_frequencies(self):
        """Get the frequency of each move.

        Returns:
            dict: Frequency counts for each move
        """
        total = sum(self.move_counts.values()) or 1
        return {move: count / total for move, count in self.move_counts.items()}

    def get_strategy_stats(self):
        """Get statistics about which strategies are being used.

        Returns:
            dict: Strategy usage and effectiveness metrics
        """
        stats = {
            "confidence": 0,
            "strategy": "None",
            "patterns": {},
            "ml_stats": {
                "randomness_ratio": round(self.randomness_used / max(1, self.total_decisions), 3),
                "longest_effective_ngram": 0
            }
        }

        predicted_move, confidence, strategy = self.predict()
        stats["confidence"] = confidence
        stats["strategy"] = strategy

        effective_ngrams = []
        for n in self.n_values:
            if len(self.history) >= n:
                context = tuple(self.history[-n:]) if n > 0 else ()
                distribution, observations = self._get_next_move_distribution(n, context)

                if distribution and observations >= 3:
                    max_prob = max(distribution.values())
                    if max_prob >= self.confidence_threshold:
                        effective_ngrams.append((n, max_prob, observations))

        if effective_ngrams:
            stats["ml_stats"]["longest_effective_ngram"] = max(effective_ngrams, key=lambda x: x[0])[0]

        pattern_strengths = []
        for n in self.n_values:
            if len(self.history) >= n:
                context = tuple(self.history[-n:]) if n > 0 else ()
                distribution, observations = self._get_next_move_distribution(n, context)

                if distribution and observations >= 2:
                    strongest_move = max(distribution, key=distribution.get)
                    if n == 0:
                        pattern_str = f"Overall → {strongest_move}"
                    else:
                        pattern_str = " → ".join(context) + f" → {strongest_move}"
                    strength = distribution[strongest_move]
                    pattern_strengths.append((pattern_str, strength, observations))

        pattern_strengths.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for i, (pattern, strength, count) in enumerate(pattern_strengths[:5]):
            stats["patterns"][pattern] = {"strength": strength, "count": count}

        return stats

    def get_ml_insights(self):
        """Get insights into how the ML model is performing.

        Returns:
            dict: ML model insights
        """
        insights = {
            "confidence_by_ngram": {},
            "effective_ngrams": [],
            "randomness_rate": self.randomness_used / max(1, self.total_decisions),
            "pattern_entropy": {}
        }

        for n in self.n_values:
            if n not in self.transitions or not self.transitions[n]:
                continue

            confidences = []
            for context, counts in self.transitions[n].items():
                total = sum(counts.values())
                if total >= 3:
                    probs = [count / total for count in counts.values()]
                    max_prob = max(probs)
                    confidences.append(max_prob)

            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                insights["confidence_by_ngram"][n] = avg_confidence

                if avg_confidence >= self.confidence_threshold:
                    insights["effective_ngrams"].append(n)

            for context, counts in self.transitions[n].items():
                total = sum(counts.values())
                if total >= 3:
                    probs = [count / total for count in counts.values()]
                    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)

                    context_str = "Overall" if not context else " → ".join(context)
                    insights["pattern_entropy"][context_str] = {
                        "entropy": entropy,
                        "predictability": 1.0 - (entropy / np.log2(3)),
                        "observations": total
                    }

        return insights


def get_predictor_path(session_id):
    """Get the path for storing predictor data."""
    predictor_dir = os.path.join(tempfile.gettempdir(), 'rps_predictors')
    os.makedirs(predictor_dir, exist_ok=True)
    return os.path.join(predictor_dir, f"predictor_{session_id}.pkl")


def save_predictor(predictor, session_id):
    """Save predictor to filesystem instead of session."""
    path = get_predictor_path(session_id)
    with open(path, 'wb') as f:
        pickle.dump(predictor, f)
    return path


def load_predictor(session_id):
    """Load predictor from filesystem."""
    path = get_predictor_path(session_id)
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        if os.path.exists(path):
            os.remove(path)
    return ImprovedMLPredictor()


@app.route('/')
def index():
    """Render the game page."""

    if 'id' not in session:
        session['id'] = base64.b64encode(os.urandom(16)).decode('utf-8')

    predictor = load_predictor(session['id'])

    if 'scores' not in session:
        session['scores'] = {'user': 0, 'ai': 0, 'tie': 0}
    if 'rounds' not in session:
        session['rounds'] = []

    return render_template('index.html',
                           scores=session['scores'],
                           rounds=session['rounds'])


@app.route('/play', methods=['POST'])
def play():
    """Handle a game play."""
    try:
        user_move = request.form.get('move')

        if user_move not in CHOICES:
            return jsonify({'error': 'Invalid move'}), 400

        predictor = load_predictor(session['id'])

        ai_move, confidence, strategy = predictor.get_counter_move()

        if user_move == ai_move:
            result = 'tie'
        elif BEATS[user_move] == ai_move:
            result = 'win'
        elif BEATS[ai_move] == user_move:
            result = 'lose'
        else:
            result = 'tie'

        scores = session['scores']
        if result == 'win':
            scores['user'] += 1
        elif result == 'lose':
            scores['ai'] += 1
        else:
            scores['tie'] += 1

        predictor.update(user_move)

        save_predictor(predictor, session['id'])

        rounds = session['rounds']

        new_round = {
            'round_number': len(rounds) + 1,
            'user_move': user_move,
            'ai_move': ai_move,
            'result': result,
            'confidence': round(confidence * 100, 1),
            'strategy': strategy
        }

        rounds.append(new_round)

        if len(rounds) > 10:
            rounds.pop(0)

            for i, r in enumerate(rounds):
                r['round_number'] = i + 1

        session['scores'] = scores
        session['rounds'] = rounds

        try:
            strategy_stats = predictor.get_strategy_stats()
            ml_insights = predictor.get_ml_insights()
        except AttributeError:
            strategy_stats = {
                "confidence": confidence,
                "strategy": strategy,
                "patterns": {}
            }
            ml_insights = {}

        return jsonify({
            'round_number': new_round['round_number'],
            'user_move': user_move,
            'ai_move': ai_move,
            'result': result,
            'confidence': round(confidence * 100, 1),
            'strategy': strategy,
            'scores': scores,
            'move_frequencies': predictor.get_move_frequencies(),
            'transitions': predictor.get_transition_heatmap(),
            'strategy_stats': strategy_stats,
            'ml_insights': ml_insights
        })
    except Exception as e:
        print(f"Error in play route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the game."""

    predictor = ImprovedMLPredictor()

    if 'id' in session:
        save_predictor(predictor, session['id'])
    else:
        session['id'] = base64.b64encode(os.urandom(16)).decode('utf-8')
        save_predictor(predictor, session['id'])

    session['scores'] = {'user': 0, 'ai': 0, 'tie': 0}
    session['rounds'] = []

    return jsonify({'message': 'Game reset successfully'})


@app.before_request
def cleanup_old_predictors():
    """Periodically clean up old predictor files."""
    if random.random() < 0.05:
        predictor_dir = os.path.join(tempfile.gettempdir(), 'rps_predictors')
        if os.path.exists(predictor_dir):
            current_time = time.time()
            for filename in os.listdir(predictor_dir):
                filepath = os.path.join(predictor_dir, filename)

                if os.path.isfile(filepath) and current_time - os.path.getmtime(filepath) > 86400:
                    try:
                        os.remove(filepath)
                    except:
                        pass


if __name__ == '__main__':
    app.run(host="0.0.0.0")
